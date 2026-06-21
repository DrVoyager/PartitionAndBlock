"""
Partition-and-block smashed-representation inversion attack for CifarNet.

This is the CifarNet counterpart to partition_blocked_smashed_attack.py. The
victim is a split CifarNet client/server pair. The attacker observes only the
noncentral smashed representation after masking the centered spatial partition
across all channels:

    observed_s = mask * victim_client(x)

The script supports the original UnSplit CifarNet split layer indices 1-6.
For the default mask, the centered protected block has side length H // 3 by
W // 3 on the smashed feature map, matching the central cell of a 3x3 spatial
partition as closely as possible for non-divisible dimensions.
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

from models import CifarNet
from util import TV, get_examples_by_class, l2loss, normalize


DEFAULT_CHECKPOINT_DIR = Path("checkpoints_cifarnet")
DEFAULT_CHECKPOINT_NAME_TEMPLATE = "cifarnet_split{split_layer}_victim.pth"
MIN_ATTACK_MODEL_ACCURACY = 50.0


@torch.no_grad()
def split_accuracy(
    client: CifarNet,
    server: CifarNet,
    dataset,
    device: torch.device,
    split_layer: int,
    batch_size: int = 128,
) -> float:
    client.eval()
    server.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = server(client(images, end=split_layer), start=split_layer + 1)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def restored_input_accuracy(
    client: CifarNet,
    server: CifarNet,
    restored_images: torch.Tensor,
    true_labels: torch.Tensor,
    device: torch.device,
    split_layer: int,
) -> Tuple[float, int, int]:
    client.eval()
    server.eval()
    logits = server(client(restored_images.to(device), end=split_layer), start=split_layer + 1)
    predicted = logits.argmax(dim=1)
    labels = true_labels.to(device)
    correct = (predicted == labels).sum().item()
    total = labels.numel()
    return 100.0 * correct / max(total, 1), correct, total


@torch.no_grad()
def clone_forward_accuracy(
    clone_client: CifarNet,
    server: CifarNet,
    dataset,
    device: torch.device,
    split_layer: int,
    batch_size: int = 128,
    max_examples: Optional[int] = None,
) -> float:
    clone_client.eval()
    server.eval()
    eval_dataset = dataset
    if max_examples is not None:
        eval_dataset = torch.utils.data.Subset(dataset, range(min(max_examples, len(dataset))))
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = server(clone_client(images, end=split_layer), start=split_layer + 1)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def train_split_cifarnet(
    client: CifarNet,
    server: CifarNet,
    trainset,
    testset,
    device: torch.device,
    split_layer: int,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> float:
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    client_opt = torch.optim.Adam(client.parameters(), lr=lr, amsgrad=True)
    server_opt = torch.optim.Adam(server.parameters(), lr=lr, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    test_acc = 0.0

    for epoch in range(epochs):
        client.train()
        server.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            client_opt.zero_grad(set_to_none=True)
            server_opt.zero_grad(set_to_none=True)
            logits = server(client(images, end=split_layer), start=split_layer + 1)
            loss = criterion(logits, labels)
            loss.backward()
            client_opt.step()
            server_opt.step()
            running_loss += loss.item()

        test_acc = split_accuracy(client, server, testset, device, split_layer, batch_size=batch_size)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"loss={running_loss / len(trainloader):.4f} | test_acc={test_acc:.2f}%",
            flush=True,
        )

    return test_acc


def save_split_checkpoint(
    checkpoint_path: Path,
    client: CifarNet,
    server: CifarNet,
    split_layer: int,
    epochs: int,
    test_acc: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epochs,
            "split_layer": split_layer,
            "client_state_dict": client.state_dict(),
            "server_state_dict": server.state_dict(),
            "test_acc": test_acc,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


def load_split_checkpoint(
    checkpoint_path: Path,
    client: CifarNet,
    server: CifarNet,
    device: torch.device,
    split_layer: int,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("split_layer") not in (None, split_layer):
        raise SystemExit(
            f"ERROR: Checkpoint split_layer={checkpoint.get('split_layer')} "
            f"does not match requested split_layer={split_layer}."
        )
    client.load_state_dict(checkpoint["client_state_dict"])
    server.load_state_dict(checkpoint["server_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")


def central_partition_bounds(s: torch.Tensor, mask_side: Optional[int] = None) -> Tuple[int, int, int, int]:
    if s.dim() != 4:
        raise ValueError(f"Expected a 4D smashed tensor [B, C, H, W], got shape {tuple(s.shape)}.")
    _, _, height, width = s.shape
    side_h = mask_side if mask_side is not None else height // 3
    side_w = mask_side if mask_side is not None else width // 3
    if side_h < 1 or side_w < 1:
        raise ValueError("Mask side must be at least 1.")
    if side_h > height or side_w > width:
        raise ValueError(
            f"Mask side {mask_side} is too large for smashed spatial shape H={height}, W={width}."
        )
    h_start = (height - side_h) // 2
    w_start = (width - side_w) // 2
    return h_start, h_start + side_h, w_start, w_start + side_w


def noncentral_smashed_mask(s: torch.Tensor, mask_side: Optional[int] = None) -> torch.Tensor:
    mask = torch.ones_like(s)
    h_start, h_end, w_start, w_end = central_partition_bounds(s, mask_side=mask_side)
    mask[:, :, h_start:h_end, w_start:w_end] = 0.0
    return mask


def masked_feature_mse(pred_s: torch.Tensor, target_s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = ((pred_s - target_s) * mask).pow(2)
    denom = mask.sum().clamp_min(1.0)
    return squared_error.sum() / denom


def blocked_smashed_inversion_attack(
    clone_client: CifarNet,
    split_layer: int,
    target_s: torch.Tensor,
    mask: torch.Tensor,
    input_size: Tuple[int, ...],
    lambda_tv: float = 0.1,
    lambda_l2: float = 0.0,
    main_iters: int = 1000,
    input_iters: int = 100,
    model_iters: int = 100,
    steal_model: bool = True,
    input_change_tol: float = 1e-7,
    main_convergence_patience: int = 5,
    min_main_iters: int = 50,
    disable_input_convergence: bool = False,
    lr_input: float = 1e-3,
    lr_model: float = 1e-3,
    clamp: bool = True,
    log_every: int = 50,
) -> Tuple[torch.Tensor, int]:
    if input_iters < 1:
        raise ValueError("input_iters must be at least 1.")
    if model_iters < 1:
        raise ValueError("model_iters must be at least 1.")
    if main_convergence_patience < 1:
        raise ValueError("main_convergence_patience must be at least 1.")
    if min_main_iters < 1:
        raise ValueError("min_main_iters must be at least 1.")

    device = target_s.device
    clone_client = clone_client.to(device)
    target_s = target_s.detach()
    mask = mask.detach()
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
            pred_s = clone_client(x_hat, end=split_layer)
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
                pred_s = clone_client(x_hat.detach(), end=split_layer)
                model_loss = masked_feature_mse(pred_s, observed_target_s, mask)
                model_loss.backward()
                model_opt.step()

        clone_client.eval()
        with torch.no_grad():
            pred_s = clone_client(x_hat, end=split_layer)
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
                f"stable_main_steps={stable_main_steps}/{main_convergence_patience}",
                flush=True,
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
                f"input_change_tol={input_change_tol:.2e}",
                flush=True,
            )
            return x_hat.detach(), main_iter + 1

    return x_hat.detach(), main_iters


def main() -> None:
    parser = argparse.ArgumentParser(description="CifarNet P&B blocked-smashed UnSplit-style attack.")
    parser.add_argument("--data-root", type=str, default="data/cifar")
    parser.add_argument("--split-layer", type=int, required=True,
                        help="Original UnSplit CifarNet split layer index. Use 1-6.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--mask-side", type=int, default=None,
                        help="Centered protected spatial mask side length. Omit to use H//3 and W//3.")
    parser.add_argument("--main-iters", type=int, default=1000)
    parser.add_argument("--input-iters", type=int, default=100)
    parser.add_argument("--model-iters", type=int, default=100)
    parser.add_argument("--lambda-tv", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--input-change-tol", "--main-loss-tol", dest="input_change_tol", type=float, default=1e-7,
                        help="Stop early when the mean absolute change in reconstructed input stays below this tolerance.")
    parser.add_argument("--main-convergence-patience", type=int, default=5)
    parser.add_argument("--min-main-iters", type=int, default=50)
    parser.add_argument("--disable-input-convergence", action="store_true")
    parser.add_argument("--known-client", action="store_true",
                        help="Use a copy of the victim client instead of learning a random clone client.")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--clone-acc-max-examples", type=int, default=2000,
                        help="Number of test samples used for clone-forward accuracy. Use 0 for the full test set.")
    args = parser.parse_args()

    if not 1 <= args.split_layer <= 6:
        raise SystemExit("ERROR: Use --split-layer 1 through 6 for CifarNet P&B blocked-smashed runs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("ERROR: CUDA is required for this run, but torch.cuda.is_available() is false.")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    trainset = datasets.CIFAR10(args.data_root, download=True, train=True, transform=transform)
    testset = datasets.CIFAR10(args.data_root, download=True, train=False, transform=transform)

    client = CifarNet().to(device)
    server = CifarNet().to(device)
    checkpoint_name = DEFAULT_CHECKPOINT_NAME_TEMPLATE.format(split_layer=args.split_layer)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(args.checkpoint_dir) / checkpoint_name

    if checkpoint_path.exists():
        load_split_checkpoint(checkpoint_path, client, server, device, args.split_layer)
    else:
        print(f"No CifarNet split checkpoint found at {checkpoint_path}; training victim split model...")
        test_acc = train_split_cifarnet(
            client,
            server,
            trainset,
            testset,
            device,
            split_layer=args.split_layer,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        save_split_checkpoint(checkpoint_path, client, server, args.split_layer, args.epochs, test_acc)

    victim_acc = split_accuracy(client, server, testset, device, args.split_layer, batch_size=args.batch_size)
    print(f"Victim split model test accuracy before attack: {victim_acc:.2f}%")
    if victim_acc < MIN_ATTACK_MODEL_ACCURACY:
        raise SystemExit(
            "ERROR: Victim model accuracy is below the attack sanity threshold: "
            f"{victim_acc:.2f}% < {MIN_ATTACK_MODEL_ACCURACY:.2f}%. Stopping before attack."
        )

    images = torch.stack([get_examples_by_class(testset, c, count=1) for c in range(10)], dim=0).to(device)
    true_labels = torch.arange(10, device=device)

    client.eval()
    with torch.no_grad():
        target_s = client(images, end=args.split_layer)
        mask = noncentral_smashed_mask(target_s, mask_side=args.mask_side)
        observed_target_s = target_s * mask

    h_start, h_end, w_start, w_end = central_partition_bounds(target_s, mask_side=args.mask_side)
    observed_ratio = mask.sum().item() / mask.numel()
    print("Victim model: CifarNet")
    print(f"Split layer: {args.split_layer}")
    print(f"Mask side: {args.mask_side if args.mask_side is not None else 'default'}")
    print(f"Full smashed representation shape: {tuple(target_s.shape)}")
    print(f"Central protected partition removed: h={h_start}:{h_end}, w={w_start}:{w_end}")
    print(f"Observed noncentral smashed representation shape: {tuple(observed_target_s.shape)}")
    print(f"Masked feature ratio: {1.0 - observed_ratio:.4f}")
    print(f"Observed feature ratio: {observed_ratio:.4f}")
    print(f"Attack regularization: lambda_tv={args.lambda_tv:g}, lambda_l2={args.lambda_l2:g}")

    if args.known_client:
        clone_client = copy.deepcopy(client).to(device)
        steal_model = False
        print("Attack mode: known client; optimizing only x_hat.")
    else:
        clone_client = CifarNet().to(device)
        steal_model = True
        print("Attack mode: unknown client; optimizing both x_hat and random clone client.")

    result, attack_iters = blocked_smashed_inversion_attack(
        clone_client=clone_client,
        split_layer=args.split_layer,
        target_s=target_s,
        mask=mask,
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
    )

    mode = "known" if args.known_client else "unknown"
    save_dir = Path(args.save_dir) if args.save_dir else Path(f"results_cifarnet_pb_{mode}_split{args.split_layer}")
    save_dir.mkdir(parents=True, exist_ok=True)

    result_to_save = normalize(result).detach().cpu()
    images_cpu = images.detach().cpu()
    mse = nn.MSELoss()
    losses = []
    for idx in range(10):
        image_loss = mse(result[idx], images[idx]).item()
        losses.append(image_loss)
        save_image(result_to_save[idx], save_dir / f"recovered_{idx}.png")
        save_image(images_cpu[idx], save_dir / f"target_{idx}.png")
        print(f"Image {idx} pixel MSE: {image_loss:.6f}")

    avg_mse = sum(losses) / len(losses)
    print(f"Average pixel MSE: {avg_mse:.6f}")
    print(f"Total attack iterations: {attack_iters}")

    restored_acc, restored_correct, restored_total = restored_input_accuracy(
        client, server, result, true_labels, device, split_layer=args.split_layer
    )
    print(
        "Restored-input clone accuracy (victim split model on restored inputs): "
        f"{restored_acc:.2f}% ({restored_correct}/{restored_total})"
    )

    clone_acc_examples = None if args.clone_acc_max_examples == 0 else args.clone_acc_max_examples
    clone_acc = clone_forward_accuracy(
        clone_client,
        server,
        testset,
        device,
        split_layer=args.split_layer,
        batch_size=args.batch_size,
        max_examples=clone_acc_examples,
    )
    print(f"Clone-forward test accuracy through victim server: {clone_acc:.4f}%")
    print(f"Saved recovered and target images to: {save_dir}")


if __name__ == "__main__":
    main()
