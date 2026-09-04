"""
Full-smashed-representation inversion attack for PartitionAndBlockingModel.

Goal:
    Given the full smashed representation
        target_s = victim.client_model(x)
    recover an input image x_hat such that
        clone_client(x_hat) ~= target_s.

This adapts the original model_inversion_stealing attack to the
PartitionAndBlockingModel, whose split point is explicitly
model.client_model rather than a generic forward(..., end=split_layer).
"""

import argparse
import copy
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from mobilenet_v1 import PartitionAndBlockingModel, MobileNetV1ClientModel
from util import TV, l2loss, normalize, get_examples_by_class


DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_CHECKPOINT_NAME = "partition_model_attack_victim.pth"


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


@torch.no_grad()
def restored_input_accuracy(
    victim_model: nn.Module,
    restored_images: torch.Tensor,
    true_labels: torch.Tensor,
    device: torch.device,
) -> Tuple[float, int, int]:
    victim_model.eval()
    logits = victim_model(restored_images.to(device))
    predicted = logits.argmax(dim=1)
    labels = true_labels.to(device)
    correct = (predicted == labels).sum().item()
    total = labels.numel()
    return 100.0 * correct / max(total, 1), correct, total


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {checkpoint_path}")


def find_existing_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None

    preferred_checkpoint = checkpoint_dir / DEFAULT_CHECKPOINT_NAME
    if preferred_checkpoint.exists():
        return preferred_checkpoint

    checkpoints = sorted(
        checkpoint_dir.glob("*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def save_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    epochs: int,
    test_acc: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "test_acc": test_acc,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


def train_partition_model(
    model: PartitionAndBlockingModel,
    trainset,
    testset,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> float:
    model.train()
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    test_acc = 0.0

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

    return test_acc


def full_smashed_inversion_attack(
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
    Recover x from the full smashed representation target_s.

    If steal_model=True, this follows the original attack structure and
    alternates between optimizing x_hat and optimizing clone_client.

    If steal_model=False, clone_client is treated as the known client model,
    and only x_hat is optimized.
    """
    device = target_s.device
    clone_client = clone_client.to(device)

    # x_hat starts as a gray image, same as the original attack.
    x_hat = torch.empty(input_size, device=device).fill_(0.5).requires_grad_(True)

    input_opt = torch.optim.Adam([x_hat], lr=lr_input, amsgrad=True)
    model_opt: Optional[torch.optim.Optimizer] = None
    if steal_model:
        model_opt = torch.optim.Adam(clone_client.parameters(), lr=lr_model, amsgrad=True)

    mse = nn.MSELoss()
    target_s = target_s.detach()

    for main_iter in range(main_iters):
        # 1) Model inversion step: update x_hat so clone_client(x_hat) matches target_s.
        clone_client.eval()
        for _ in range(input_iters):
            input_opt.zero_grad(set_to_none=True)
            pred_s = clone_client(x_hat)
            loss = mse(pred_s, target_s) + lambda_tv * TV(x_hat) + lambda_l2 * l2loss(x_hat)
            loss.backward()
            input_opt.step()

            if clamp:
                with torch.no_grad():
                    x_hat.clamp_(0.0, 1.0)

        # 2) Model stealing step: update clone_client so it maps x_hat to target_s.
        if steal_model:
            clone_client.train()
            for _ in range(model_iters):
                assert model_opt is not None
                model_opt.zero_grad(set_to_none=True)
                pred_s = clone_client(x_hat.detach())
                model_loss = mse(pred_s, target_s)
                model_loss.backward()
                model_opt.step()

        if log_every and ((main_iter + 1) % log_every == 0 or main_iter == 0):
            clone_client.eval()
            with torch.no_grad():
                match_loss = mse(clone_client(x_hat), target_s).item()
            print(f"iter {main_iter + 1:04d}/{main_iters} | smashed_mse={match_loss:.6f}")

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
    parser.add_argument("--save-dir", type=str, default="results_partition_attack")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint for a trained PartitionAndBlockingModel.")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR),
                        help="Directory to search for or save trained victim weights.")
    parser.add_argument("--protected-width", type=float, default=1.0,
                        help="Width multiplier for the protected convolutional encoder (default: 1.0).")
    parser.add_argument("--protected-pool-size", type=int, default=2,
                        help="Adaptive pooling side length for the protected encoder (default: 2).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    trainset = datasets.CIFAR10(args.data_root, download=True, train=True, transform=transform)
    testset = datasets.CIFAR10(args.data_root, download=True, train=False, transform=transform)

    victim = PartitionAndBlockingModel(
        num_classes=args.num_classes,
        protected_width=args.protected_width,
        protected_pool_size=args.protected_pool_size,
    ).to(device)

    if args.checkpoint:
        load_model_checkpoint(victim, Path(args.checkpoint), device)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        existing_checkpoint = find_existing_checkpoint(checkpoint_dir)
        if existing_checkpoint is not None:
            load_model_checkpoint(victim, existing_checkpoint, device)
        else:
            print(f"No weights found in {checkpoint_dir}; training victim PartitionAndBlockingModel...")
            test_acc = train_partition_model(victim, trainset, testset, device, epochs=args.epochs, batch_size=args.batch_size)
            save_model_checkpoint(victim, checkpoint_dir / DEFAULT_CHECKPOINT_NAME, args.epochs, test_acc)

    victim.eval()

    # One target image per CIFAR-10 class.
    images = torch.stack([get_examples_by_class(testset, c, count=1) for c in range(args.num_classes)], dim=0).to(device)
    true_labels = torch.arange(args.num_classes, device=device)

    # Full smashed representation: this is the attack observation.
    with torch.no_grad():
        target_s = victim.client_model(images)

    print(f"Target full smashed representation shape: {tuple(target_s.shape)}")

    if args.known_client:
        # Stronger attacker: knows the victim client architecture and weights.
        clone_client = copy.deepcopy(victim.client_model).to(device)
        steal_model = False
        print("Attack mode: known client; optimizing only x_hat.")
    else:
        # Same spirit as the original model_inversion_stealing attack:
        # attacker starts with a fresh clone client and optimizes both x_hat and clone_client.
        clone_client = MobileNetV1ClientModel().to(device)
        steal_model = True
        print("Attack mode: random clone client; optimizing both x_hat and clone_client.")

    result = full_smashed_inversion_attack(
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

    # CIFAR outputs are already clamped during optimization, but normalize is useful
    # for viewing if clamp is disabled or changed later.
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
    restored_acc, restored_correct, restored_total = restored_input_accuracy(victim, result, true_labels, device)
    print(
        "Clone accuracy (victim model on restored inputs): "
        f"{restored_acc:.2f}% ({restored_correct}/{restored_total})"
    )
    print(f"Saved recovered and target images to: {save_dir}")


if __name__ == "__main__":
    main()
