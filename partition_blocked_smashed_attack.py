"""
Blocked-smashed-representation inversion attack for PartitionAndBlockingModel.

The attacker observes only the noncentral smashed representation:

    target_blocked_s = mask * victim.client_model(x)

and recovers x_hat by matching only the unmasked smashed features. With the
default stem split, victim.client_model(x) has shape [B, 32, 32, 32] and the
protected central partition is h=11:21, w=11:21. Use --split-after-depthwise N
to move the split after the Nth MobileNetV1 depthwise block.
"""

import argparse
import copy
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from mobilenet_v1 import (
    DEPTHWISE_BLOCKS,
    SPLIT_PRESETS,
    MobileNetV1ClientModel,
    PartitionAndBlockingModel,
    split_children_from_depthwise_blocks,
)
from util import TV, get_examples_by_class, l2loss, normalize


MIN_ATTACK_MODEL_ACCURACY = 50.0
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def cifar10_normalize_tensor(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (images - mean) / std


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
    input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[float, int, int]:
    victim_model.eval()
    model_input = restored_images.to(device)
    if input_transform is not None:
        model_input = input_transform(model_input)
    logits = victim_model(model_input)
    predicted = logits.argmax(dim=1)
    labels = true_labels.to(device)
    correct = (predicted == labels).sum().item()
    total = labels.numel()
    return 100.0 * correct / max(total, 1), correct, total


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


def load_partition_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    split_children: int,
    mask_side: Optional[int],
    protected_width: float,
    protected_pool_size: int,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise SystemExit(
            f"ERROR: Checkpoint {checkpoint_path} is incompatible with split_children={split_children}, "
            f"mask_side={mask_side}, protected_width={protected_width}, and "
            f"protected_pool_size={protected_pool_size}. Retrain the model for this configuration "
            "or provide a matching checkpoint."
        ) from exc
    print(f"Loaded checkpoint: {checkpoint_path}")


def central_partition_bounds(model: PartitionAndBlockingModel, s: torch.Tensor) -> Tuple[int, int, int, int]:
    return model.central_partition_bounds(s)


def noncentral_smashed_mask(model: PartitionAndBlockingModel, s: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(s)
    h_start, h_end, w_start, w_end = central_partition_bounds(model, s)
    mask[:, :, h_start:h_end, w_start:w_end] = 0.0
    return mask


def masked_feature_mse(pred_s: torch.Tensor, target_s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = ((pred_s - target_s) * mask).pow(2)
    denom = mask.sum().clamp_min(1.0)
    return squared_error.sum() / denom


def blocked_smashed_inversion_attack(
    partition_model: PartitionAndBlockingModel,
    clone_client: nn.Module,
    target_s: torch.Tensor,
    input_size: Tuple[int, ...],
    lambda_tv: float = 0.1,
    lambda_l2: float = 0.0,
    main_iters: int = 1000,
    input_iters: int = 100,
    model_iters: int = 100,
    lr_input: float = 1e-3,
    lr_model: float = 1e-3,
    steal_model: bool = True,
    clamp: bool = True,
    log_every: int = 50,
    input_change_tol: float = 1e-7,
    main_convergence_patience: int = 5,
    min_main_iters: int = 50,
    disable_input_convergence: bool = False,
    input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    if input_iters < 1:
        raise ValueError("input_iters must be at least 1.")
    if model_iters < 1:
        raise ValueError("model_iters must be at least 1.")
    if main_convergence_patience < 1:
        raise ValueError("main_convergence_patience must be at least 1.")
    if min_main_iters < 1:
        raise ValueError("min_main_iters must be at least 1.")

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

    previous_x_hat = x_hat.detach().clone()
    stable_main_steps = 0

    for main_iter in range(main_iters):
        clone_client.eval()
        for _ in range(input_iters):
            input_opt.zero_grad(set_to_none=True)
            model_input = input_transform(x_hat) if input_transform is not None else x_hat
            pred_s = clone_client(model_input)
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
                detached_x = x_hat.detach()
                model_input = input_transform(detached_x) if input_transform is not None else detached_x
                pred_s = clone_client(model_input)
                model_loss = masked_feature_mse(pred_s, observed_target_s, mask)
                model_loss.backward()
                model_opt.step()

        clone_client.eval()
        with torch.no_grad():
            model_input = input_transform(x_hat) if input_transform is not None else x_hat
            pred_s = clone_client(model_input)
            noncentral_loss = masked_feature_mse(pred_s, observed_target_s, mask).item()
            input_change = (x_hat.detach() - previous_x_hat).abs().mean().item()

        if input_change <= input_change_tol:
            stable_main_steps += 1
        else:
            stable_main_steps = 0
        previous_x_hat = x_hat.detach().clone()

        if log_every and ((main_iter + 1) % log_every == 0 or main_iter == 0):
            print(
                f"iter {main_iter + 1:04d}/{main_iters} | "
                f"noncentral_smashed_mse={noncentral_loss:.6f} | "
                f"mean_abs_input_change={input_change:.8f} | "
                f"stable_main_steps={stable_main_steps}/{main_convergence_patience}"
            )

        if (
            not disable_input_convergence
            and main_iter + 1 >= min_main_iters
            and stable_main_steps >= main_convergence_patience
        ):
            print(
                f"Converged at iter {main_iter + 1:04d}/{main_iters}: "
                f"noncentral_smashed_mse={noncentral_loss:.6f}, "
                f"mean_abs_input_change={input_change:.8f}, "
                f"stable_main_steps={stable_main_steps}/{main_convergence_patience}, "
                f"input_change_tol={input_change_tol:.2e}"
            )
            break

    return x_hat.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/cifar")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--split-preset", choices=tuple(SPLIT_PRESETS), default="stem",
                        help="Named MobileNetV1 split point. stem means before the first depthwise block; dwN means after the Nth depthwise block.")
    parser.add_argument("--split-after-depthwise", type=int, default=None,
                        help=f"Split after the Nth MobileNetV1 depthwise block, from 1 to {DEPTHWISE_BLOCKS}.")
    parser.add_argument("--split-children", type=int, default=None,
                        help="Low-level override: number of spatial MobileNetV1 feature children in the P&B client.")
    parser.add_argument("--mask-side", type=int, default=None,
                        help="Override the centered protected spatial mask side length. Omit to use the default split-dependent size.")
    parser.add_argument("--protected-width", type=float, default=1.0,
                        help="Width multiplier for the protected convolutional encoder (default: 1.0).")
    parser.add_argument("--protected-pool-size", type=int, default=2,
                        help="Adaptive pooling side length for the protected encoder (default: 2).")
    parser.add_argument("--main-iters", type=int, default=1000)
    parser.add_argument("--input-iters", type=int, default=100)
    parser.add_argument("--model-iters", type=int, default=100)
    parser.add_argument("--lambda-tv", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--input-change-tol", "--main-loss-tol", dest="input_change_tol", type=float, default=1e-7,
                        help="Stop early when the mean absolute change in reconstructed input stays below this tolerance.")
    parser.add_argument("--main-convergence-patience", type=int, default=5,
                        help="Number of consecutive stable main iterations required for convergence.")
    parser.add_argument("--min-main-iters", type=int, default=50,
                        help="Minimum number of main iterations before convergence can stop the attack.")
    parser.add_argument("--disable-input-convergence", action="store_true",
                        help="Ignore reconstructed-input convergence and run until main-iters is reached.")
    parser.add_argument("--known-client", action="store_true",
                        help="Use a copy of the victim client instead of stealing a random clone.")
    parser.add_argument("--save-dir", type=str, default="results_partition_blocked_attack")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint for a trained PartitionAndBlockingModel.")
    parser.add_argument("--require-cuda", action="store_true", help="Stop before running if CUDA is not available.")
    args = parser.parse_args()

    if args.split_after_depthwise is not None:
        args.split_children = split_children_from_depthwise_blocks(args.split_after_depthwise)
    elif args.split_children is None:
        args.split_children = SPLIT_PRESETS[args.split_preset]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("ERROR: CUDA is required for this run, but torch.cuda.is_available() is false.")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    trainset = datasets.CIFAR10(args.data_root, download=True, train=True, transform=transform)
    testset = datasets.CIFAR10(args.data_root, download=True, train=False, transform=transform)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_testset = datasets.CIFAR10(args.data_root, download=True, train=False, transform=eval_transform)

    victim = PartitionAndBlockingModel(
        num_classes=args.num_classes,
        split_children=args.split_children,
        mask_side=args.mask_side,
        protected_width=args.protected_width,
        protected_pool_size=args.protected_pool_size,
    ).to(device)

    if args.checkpoint:
        load_partition_checkpoint(
            victim,
            Path(args.checkpoint),
            device,
            args.split_children,
            args.mask_side,
            args.protected_width,
            args.protected_pool_size,
        )
    else:
        print(
            "Training victim PartitionAndBlockingModel "
            f"for split_children={args.split_children}. "
            "Use a matching checkpoint for this split to skip retraining."
        )
        train_partition_model(victim, trainset, testset, device, epochs=args.epochs, batch_size=args.batch_size)

    victim.eval()
    victim_acc = accuracy(victim, eval_testset, device, batch_size=args.batch_size)
    print(f"Victim model test accuracy before attack: {victim_acc:.2f}%")
    if victim_acc < MIN_ATTACK_MODEL_ACCURACY:
        raise SystemExit(
            "ERROR: Victim model accuracy is below the attack sanity threshold: "
            f"{victim_acc:.2f}% < {MIN_ATTACK_MODEL_ACCURACY:.2f}%. "
            "Stopping before attack."
        )

    images = torch.stack([get_examples_by_class(testset, c, count=1) for c in range(args.num_classes)], dim=0).to(device)
    true_labels = torch.arange(args.num_classes, device=device)

    with torch.no_grad():
        target_s = victim.client_model(cifar10_normalize_tensor(images))
        mask = noncentral_smashed_mask(victim, target_s)
        observed_target_s = target_s * mask

    h_start, h_end, w_start, w_end = central_partition_bounds(victim, target_s)
    print("Victim model: P&B MobileNet V1")
    print(f"Split preset: {args.split_preset}")
    if args.split_after_depthwise is not None:
        print(f"Split after depthwise block: {args.split_after_depthwise}")
    print(f"Client split children: {args.split_children}")
    print(f"Mask side: {args.mask_side if args.mask_side is not None else 'default'}")
    print(f"Protected encoder width: {args.protected_width}")
    print(f"Protected encoder pool size: {args.protected_pool_size}")
    print(f"Full smashed representation shape: {tuple(target_s.shape)}")
    print(f"Central protected partition removed: h={h_start}:{h_end}, w={w_start}:{w_end}")
    print(f"Observed noncentral smashed representation shape: {tuple(observed_target_s.shape)}")
    observed_ratio = mask.sum().item() / mask.numel()
    print(f"Masked feature ratio: {1.0 - observed_ratio:.4f}")
    print(f"Observed feature ratio: {observed_ratio:.4f}")
    print(f"Attack regularization: lambda_tv={args.lambda_tv:g}, lambda_l2={args.lambda_l2:g}")

    if args.known_client:
        clone_client = copy.deepcopy(victim.client_model).to(device)
        steal_model = False
        print("Attack mode: known client; optimizing only x_hat.")
    else:
        clone_client = MobileNetV1ClientModel(split_children=args.split_children).to(device)
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
        input_change_tol=args.input_change_tol,
        main_convergence_patience=args.main_convergence_patience,
        min_main_iters=args.min_main_iters,
        disable_input_convergence=args.disable_input_convergence,
        input_transform=cifar10_normalize_tensor,
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
    restored_acc, restored_correct, restored_total = restored_input_accuracy(
        victim,
        result,
        true_labels,
        device,
        input_transform=cifar10_normalize_tensor,
    )
    print(
        "Restored-input clone accuracy (victim model on restored inputs): "
        f"{restored_acc:.2f}% ({restored_correct}/{restored_total})"
    )
    print(f"Saved recovered and target images to: {save_dir}")


if __name__ == "__main__":
    main()
