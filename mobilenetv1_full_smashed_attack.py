"""
Full-smashed-representation inversion attack for baseline MobileNetV1.

By default this uses the original split before the first MobileNetV1 depthwise
separable block, i.e. the stem output [B, 32, 32, 32]. Use
--split-after-depthwise N to attack after the Nth depthwise block.
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

from util import TV, get_examples_by_class, l2loss, normalize


DEFAULT_CHECKPOINT_DIR = Path("checkpoints_mobilenetv1")
DEFAULT_CHECKPOINT_NAME = "mobilenetv1_attack_victim.pth"
MIN_ATTACK_MODEL_ACCURACY = 50.0
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
STEM_CHILDREN = 3
DEPTHWISE_BLOCKS = 12
SPLIT_PRESETS = {"stem": STEM_CHILDREN}
SPLIT_PRESETS.update({f"dw{idx}": STEM_CHILDREN + idx for idx in range(1, DEPTHWISE_BLOCKS + 1)})
SPLIT_PRESETS.update({
    "early64": SPLIT_PRESETS["dw1"],
    "pb-spatial": SPLIT_PRESETS["dw2"],
})


def split_children_from_depthwise_blocks(block_count: int) -> int:
    if not 1 <= block_count <= DEPTHWISE_BLOCKS:
        raise ValueError(f"split-after-depthwise must be between 1 and {DEPTHWISE_BLOCKS}.")
    return STEM_CHILDREN + block_count


def cifar10_normalize_tensor(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (images - mean) / std


class MobileNetV1(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            self._make_depthwise(32, 64, stride=1),
            self._make_depthwise(64, 128, stride=2),
            self._make_depthwise(128, 128, stride=1),
            self._make_depthwise(128, 256, stride=2),
            self._make_depthwise(256, 256, stride=1),
            self._make_depthwise(256, 512, stride=2),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 1024, stride=2),
            self._make_depthwise(1024, 1024, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )

    def _make_depthwise(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MobileNetV1ClientModel(nn.Module):
    def __init__(self, mobilenet: MobileNetV1, split_children: int = 3):
        super().__init__()
        if not 1 <= split_children <= len(mobilenet.features):
            raise ValueError(f"split_children must be between 1 and {len(mobilenet.features)}.")
        self.features = nn.Sequential(*list(mobilenet.features.children())[:split_children])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


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


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {checkpoint_path}")


def find_existing_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None

    for checkpoint_name in (DEFAULT_CHECKPOINT_NAME, "baseline_mobilenetv1.pth"):
        checkpoint_path = checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            return checkpoint_path

    checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def save_model_checkpoint(model: nn.Module, checkpoint_path: Path, epochs: int, test_acc: float) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epochs, "model_state_dict": model.state_dict(), "test_acc": test_acc}, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def train_mobilenetv1(
    model: MobileNetV1,
    trainset,
    testset,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> float:
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
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        test_acc = accuracy(model, testset, device)
        print(f"Epoch {epoch + 1:02d}/{epochs} | loss={running_loss / len(loader):.4f} | test_acc={test_acc:.2f}%")

    return test_acc


def full_smashed_inversion_attack(
    clone_client: nn.Module,
    target_s: torch.Tensor,
    input_size: Tuple[int, ...],
    lambda_tv: float = 0.1,
    lambda_l2: float = 0.0,
    main_iters: int = 1000,
    input_iters: int = 100,
    model_iters: int = 100,
    input_change_tol: float = 1e-7,
    main_convergence_patience: int = 5,
    min_main_iters: int = 50,
    disable_input_convergence: bool = False,
    input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    lr_input: float = 1e-3,
    lr_model: float = 1e-3,
    steal_model: bool = True,
    clamp: bool = True,
    log_every: int = 50,
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
    clone_client = clone_client.to(device)
    x_hat = torch.empty(input_size, device=device).fill_(0.5).requires_grad_(True)
    input_opt = torch.optim.Adam([x_hat], lr=lr_input, amsgrad=True)
    model_opt: Optional[torch.optim.Optimizer] = None
    if steal_model:
        model_opt = torch.optim.Adam(clone_client.parameters(), lr=lr_model, amsgrad=True)

    mse = nn.MSELoss()
    target_s = target_s.detach()
    previous_x_hat = x_hat.detach().clone()
    stable_main_steps = 0

    for main_iter in range(main_iters):
        clone_client.eval()
        for _ in range(input_iters):
            input_opt.zero_grad(set_to_none=True)
            model_input = input_transform(x_hat) if input_transform is not None else x_hat
            loss = mse(clone_client(model_input), target_s) + lambda_tv * TV(x_hat) + lambda_l2 * l2loss(x_hat)
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
                model_loss = mse(clone_client(model_input), target_s)
                model_loss.backward()
                model_opt.step()

        clone_client.eval()
        with torch.no_grad():
            model_input = input_transform(x_hat) if input_transform is not None else x_hat
            match_loss = mse(clone_client(model_input), target_s).item()
            input_change = (x_hat.detach() - previous_x_hat).abs().mean().item()

        if input_change <= input_change_tol:
            stable_main_steps += 1
        else:
            stable_main_steps = 0
        previous_x_hat = x_hat.detach().clone()

        if log_every and ((main_iter + 1) % log_every == 0 or main_iter == 0):
            print(
                f"iter {main_iter + 1:04d}/{main_iters} | "
                f"smashed_mse={match_loss:.6f} | "
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
                f"smashed_mse={match_loss:.6f}, "
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
                        help="Low-level override: number of MobileNetV1 feature children in the client.")
    parser.add_argument("--main-iters", type=int, default=1000)
    parser.add_argument("--input-iters", type=int, default=100)
    parser.add_argument("--model-iters", type=int, default=100)
    parser.add_argument("--input-change-tol", "--input-loss-tol", dest="input_change_tol", type=float, default=1e-7)
    parser.add_argument("--main-convergence-patience", "--convergence-patience", dest="main_convergence_patience", type=int, default=5)
    parser.add_argument("--min-main-iters", type=int, default=50)
    parser.add_argument("--disable-input-convergence", action="store_true")
    parser.add_argument("--lambda-tv", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--known-client", action="store_true")
    parser.add_argument("--save-dir", type=str, default="results_mobilenetv1_full_smashed_attack")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--require-cuda", action="store_true")
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

    victim = MobileNetV1(num_classes=args.num_classes).to(device)

    if args.checkpoint:
        load_model_checkpoint(victim, Path(args.checkpoint), device)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        existing_checkpoint = find_existing_checkpoint(checkpoint_dir)
        if existing_checkpoint is not None:
            load_model_checkpoint(victim, existing_checkpoint, device)
        else:
            print(f"No MobileNetV1 weights found in {checkpoint_dir}; training victim model...")
            test_acc = train_mobilenetv1(victim, trainset, testset, device, epochs=args.epochs, batch_size=args.batch_size)
            save_model_checkpoint(victim, checkpoint_dir / DEFAULT_CHECKPOINT_NAME, args.epochs, test_acc)

    victim.eval()
    victim_acc = accuracy(victim, eval_testset, device, batch_size=args.batch_size)
    print(f"Victim model test accuracy before attack: {victim_acc:.2f}%")
    if victim_acc < MIN_ATTACK_MODEL_ACCURACY:
        raise SystemExit(
            "ERROR: Victim model accuracy is below the attack sanity threshold: "
            f"{victim_acc:.2f}% < {MIN_ATTACK_MODEL_ACCURACY:.2f}%. "
            "Stopping before attack."
        )

    victim_client = MobileNetV1ClientModel(victim, split_children=args.split_children).to(device)
    victim_client.eval()

    images = torch.stack([get_examples_by_class(testset, c, count=1) for c in range(args.num_classes)], dim=0).to(device)
    true_labels = torch.arange(args.num_classes, device=device)

    with torch.no_grad():
        target_s = victim_client(cifar10_normalize_tensor(images))

    print("Victim model: MobileNet V1")
    print(f"Split preset: {args.split_preset}")
    if args.split_after_depthwise is not None:
        print(f"Split after depthwise block: {args.split_after_depthwise}")
    print(f"Client split children: {args.split_children}")
    print(f"Target full smashed representation shape: {tuple(target_s.shape)}")
    print(f"Attack regularization: lambda_tv={args.lambda_tv:g}, lambda_l2={args.lambda_l2:g}")

    if args.known_client:
        clone_client = copy.deepcopy(victim_client).to(device)
        steal_model = False
        print("Attack mode: known client; optimizing only x_hat.")
    else:
        random_victim = MobileNetV1(num_classes=args.num_classes).to(device)
        clone_client = MobileNetV1ClientModel(random_victim, split_children=args.split_children).to(device)
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
        input_change_tol=args.input_change_tol,
        main_convergence_patience=args.main_convergence_patience,
        min_main_iters=args.min_main_iters,
        disable_input_convergence=args.disable_input_convergence,
        input_transform=cifar10_normalize_tensor,
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
