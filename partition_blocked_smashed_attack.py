"""
Blocked-smashed-representation inversion attack for PartitionAndBlockingModel.

Goal:
    Given only the smashed representation outside the central protected partition,
        target_blocked_s = mask * victim.client_model(x)
    recover an input image x_hat such that
        mask * clone_client(x_hat) ~= target_blocked_s.

This is different from attacking the full smashed representation. The central
protected partition is excluded from the matching loss, so the attacker receives
no reconstruction signal from that protected region of the smashed tensor.
"""

import argparse
import copy
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from mobilenet_v1 import PartitionAndBlockingModel, MobileNetV1ClientModel
from util import TV, l2loss, normalize, get_examples_by_class


@torch.no_grad()
def accuracy(model: nn.Module, dataset, device: torch.device, batch_size: int = 128) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def train_partition_model(
    model: PartitionAndBlockingModel,
    trainset,
    testset,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> None:
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_acc = accuracy(model, testset, device)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"loss={running_loss / len(loader):.4f} | test_acc={test_acc:.2f}%"
        )


def central_partition_bounds(model: PartitionAndBlockingModel, s: torch.Tensor) -> Tuple[int, int, int, int]:
    """Return the spatial bounds of the central protected partition in s."""
    _, _, height, width = s.shape
    grid_h = model.grid_h
    grid_w = model.grid_w

    # Same central cell as PartitionAndBlockingModel.extract_central_partition: i = 1, j = 1.
    h_start = grid_h
    h_end = min(h_start + grid_h, height)
    w_start = grid_w
    w_end = min(w_start + grid_w, width)
    return h_start, h_end, w_start, w_end


def noncentral_smashed_mask(model: PartitionAndBlockingModel, s: torch.Tensor) -> torch.Tensor:
    """
    Build a binary mask with 0 on the central protected partition and 1 elsewhere.

    If s has shape [B, C, H, W], the returned mask has the same shape and can be
    multiplied with s or clone_client(x_hat).
    """
    mask = torch.ones_like(s)
    h_start, h_end, w_start, w_end = central_partition_bounds(model, s)
    mask[:, :, h_start:h_end, w_start:w_end] = 0.0
    return mask


def masked_feature_mse(pred_s: torch.Tensor, target_s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    MSE over observed noncentral smashed locations only.

    This normalizes by the number of unmasked elements instead of by all elements,
    so the loss scale does not shrink just because the protected region is removed.
    """
    squared_error = ((pred_s - target_s) * mask).pow(2)
    denom = mask.sum().clamp_min(1.0)
    return squared_error.sum() / denom


def blocked_smashed_inversion_attack(
    partition_model: PartitionAndBlockingModel,
    clone_client: nn.Module,
    target_s: torch.Tensor,
    input_size: Tuple[int, ...],
    lambda_tv: float = 0.1,
    lambda_l2: float = 1.0,
    main_iters: int = 1000,
    input_iters: int = 100,
    model_iters: int = 100,
    lr_input: float = 1e-3,
    lr_model: float = 1e-3,
    steal_model: bool = True,
    clamp: bool = True,
    log_every: int = 50,
) -> torch.Tensor:
    """
    Recover x from the noncentral part of the smashed representation.

    The attack observation is equivalent to target_s with the central protected
    partition masked out. The optimization loss compares only the noncentral
    locations:

        loss_match = MSE(mask * clone_client(x_hat), mask * target_s)

    If steal_model=True, this follows the original inversion + stealing pattern
    and alternates between optimizing x_hat and optimizing clone_client.
    If steal_model=False, clone_client is treated as a known victim client and
    only x_hat is optimized.
    """
    device = target_s.device
    partition_model = partition_model.to(device)
    clone_client = clone_client.to(device)

    target_s = target_s.detach()
    mask = noncentral_smashed_mask(partition_model, target_s).detach()
    observed_target_s = target_s * mask

    x_hat = torch.empty(input_size, device=device).fill_(0.5).requires_grad_(True)

    input_opt = torch.optim.Adam([x_hat], lr=lr_input, amsgrad=True)
    model_opt: Optional[torch.optim.Optimizer] = None
    if steal_model:
        model_opt = torch.optim.Adam(clone_client.parameters(), lr=lr_model, amsgrad=True)

    for main_iter in range(main_iters):
        clone_client.eval()
        for _ in range(input_iters):
            input_opt.zero_grad(set_to_none=True)
            pred_s = clone_client(x_hat)
            match_loss = masked_feature_mse(pred_s, observed_target_s, mask)
            loss = match_loss + lambda_tv * TV(x_hat) + lambda_l2 * l2loss(x_hat)
            loss.backward()
            input_opt.step()

            if clamp:
                with torch.no_grad():
                    x_hat.clamp_(0.0, 1.0)

        if steal_model:
            clone_client.train()
            for _ in range(model_iters):
                assert model_opt is not None
                model_opt.zero_grad(set_to_none=True)
                pred_s = clone_client(x_hat.detach())
                model_loss = masked_feature_mse(pred_s, observed_target_s, mask)
                model_loss.backward()
                model_opt.step()

        if log_every and ((main_iter + 1) % log_every == 0 or main_iter == 0):
            clone_client.eval()
            with torch.no_grad():
                pred_s = clone_client(x_hat)
                noncentral_loss = masked_feature_mse(pred_s, observed_target_s, mask).item()
            print(
                f"iter {main_iter + 1:04d}/{main_iters} | "
                f"noncentral_smashed_mse={noncentral_loss:.6f}"
            )

    return x_hat.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/cifar")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--main-iters", type=int, default=1000)
    parser.add_argument("--input-iters", type=int, default=100)
    parser.add_argument("--model-iters", type=int, default=100)
    parser.add_argument("--lambda-tv", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--known-client", action="store_true", help="Use a copy of the victim client instead of stealing a random clone.")
    parser.add_argument("--save-dir", type=str, default="results_partition_blocked_attack")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint for a trained PartitionAndBlockingModel.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    trainset = datasets.CIFAR10(args.data_root, download=True, train=True, transform=transform)
    testset = datasets.CIFAR10(args.data_root, download=True, train=False, transform=transform)

    victim = PartitionAndBlockingModel(num_classes=args.num_classes).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            victim.load_state_dict(checkpoint["model_state_dict"])
        else:
            victim.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Training victim PartitionAndBlockingModel...")
        train_partition_model(victim, trainset, testset, device, epochs=args.epochs, batch_size=args.batch_size)

    victim.eval()

    images = torch.stack([get_examples_by_class(testset, c, count=1) for c in range(args.num_classes)], dim=0).to(device)

    # Full smashed representation is computed internally, but the attacker observes
    # only the noncentral/blocked version used as the matching target.
    with torch.no_grad():
        target_s = victim.client_model(images)
        mask = noncentral_smashed_mask(victim, target_s)
        observed_target_s = target_s * mask

    h_start, h_end, w_start, w_end = central_partition_bounds(victim, target_s)
    print(f"Full smashed representation shape: {tuple(target_s.shape)}")
    print(f"Central protected partition removed: h={h_start}:{h_end}, w={w_start}:{w_end}")
    print(f"Observed noncentral smashed representation shape: {tuple(observed_target_s.shape)}")
    print(f"Observed feature ratio: {mask.sum().item() / mask.numel():.4f}")

    if args.known_client:
        clone_client = copy.deepcopy(victim.client_model).to(device)
        steal_model = False
        print("Attack mode: known client; optimizing only x_hat.")
    else:
        clone_client = MobileNetV1ClientModel().to(device)
        steal_model = True
        print("Attack mode: random clone client; optimizing both x_hat and clone_client.")

    result = blocked_smashed_inversion_attack(
        partition_model=victim,
        clone_client=clone_client,
        target_s=target_s,
        input_size=tuple(images.size()),
        lambda_tv=args.lambda_tv,
        lambda_l2=args.lambda_l2,
        main_iters=args.main_iters,
        input_iters=args.input_iters,
        model_iters=args.model_iters,
        steal_model=steal_model,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    result_to_save = normalize(result).detach().cpu()
    images_cpu = images.detach().cpu()

    mse = nn.MSELoss()
    losses = []
    for idx in range(args.num_classes):
        image_loss = mse(result[idx], images[idx]).item()
        losses.append(image_loss)
        save_image(result_to_save[idx], save_dir / f"recovered_{idx}.png")
        save_image(images_cpu[idx], save_dir / f"target_{idx}.png")
        print(f"Image {idx} pixel MSE: {image_loss:.6f}")

    print(f"Average pixel MSE: {sum(losses) / len(losses):.6f}")
    print(f"Saved recovered and target images to: {save_dir}")


if __name__ == "__main__":
    main()
