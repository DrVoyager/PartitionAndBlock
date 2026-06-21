"""
Full-smashed-representation inversion attack for standard CIFAR-10 classifiers.

Goal:
    Given the full smashed representation
        target_s = victim_client(x)
    recover an input image x_hat such that
        clone_client(x_hat) ~= target_s.

This is based on partition_full_smashed_attack.py, but replaces the
PartitionAndBlockingModel victim with a standard classifier such as
MobileNet V1 or the original unsplit CifarNet.
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

from models import CifarNet
from util import TV, l2loss, normalize, get_examples_by_class


DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_ATTACK_STATE_NAME = "attack_state.pt"
MIN_ATTACK_MODEL_ACCURACY = 50.0
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
MODEL_CHECKPOINT_NAMES = {
    "mobilenetv1": "mobilenetv1_attack_victim.pth",
    "cifarnet": "cifarnet_attack_victim.pth",
}


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
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class MobileNetV1ClientModel(nn.Module):
    """Client-side MobileNet V1 prefix used to produce the smashed representation."""

    def __init__(self, mobilenet: MobileNetV1, split_children: int = 3):
        super().__init__()
        if not 1 <= split_children <= len(mobilenet.features):
            raise ValueError(f"split_children must be between 1 and {len(mobilenet.features)}.")
        self.features = nn.Sequential(*list(mobilenet.features.children())[:split_children])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class CifarNetClientModel(nn.Module):
    """Client-side CifarNet prefix using the original unsplit split layer index."""

    def __init__(self, cifarnet: CifarNet, split_layer: int = 1):
        super().__init__()
        if not 0 <= split_layer <= 17:
            raise ValueError("CifarNet split_layer must be between 0 and 17.")
        self.cifarnet = cifarnet
        self.split_layer = split_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cifarnet(x, end=self.split_layer)


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
    return find_existing_checkpoint_for_model(checkpoint_dir, MODEL_CHECKPOINT_NAMES["mobilenetv1"])


def find_existing_checkpoint_for_model(checkpoint_dir: Path, checkpoint_name: str) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None

    preferred_checkpoint = checkpoint_dir / checkpoint_name
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


def train_classifier_model(
    model: nn.Module,
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


def build_victim_model(victim_model: str, num_classes: int) -> nn.Module:
    if victim_model == "mobilenetv1":
        return MobileNetV1(num_classes=num_classes)
    if victim_model == "cifarnet":
        if num_classes != 10:
            raise ValueError("CifarNet is fixed to 10 CIFAR-10 classes.")
        return CifarNet()
    raise ValueError(f"Unsupported victim model: {victim_model}")


def build_client_model(
    victim: nn.Module,
    victim_model: str,
    split_children: int,
    split_layer: int,
) -> nn.Module:
    if victim_model == "mobilenetv1":
        if not isinstance(victim, MobileNetV1):
            raise TypeError("Expected MobileNetV1 victim.")
        return MobileNetV1ClientModel(victim, split_children=split_children)
    if victim_model == "cifarnet":
        if not isinstance(victim, CifarNet):
            raise TypeError("Expected CifarNet victim.")
        return CifarNetClientModel(victim, split_layer=split_layer)
    raise ValueError(f"Unsupported victim model: {victim_model}")


def format_model_name(victim_model: str) -> str:
    if victim_model == "mobilenetv1":
        return "MobileNet V1"
    if victim_model == "cifarnet":
        return "CifarNet"
    return victim_model


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
    state_path: Optional[Path] = None,
    reset_attack_state: bool = False,
    state_save_every: int = 50,
    attack_metadata: Optional[dict] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Recover x from the full smashed representation target_s.

    If steal_model=True, alternates between optimizing x_hat and clone_client.
    If steal_model=False, clone_client is known and only x_hat is optimized.
    """
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
    completed_iters = 0
    input_opt_state = None
    model_opt_state = None

    if state_path is not None and state_path.exists() and not reset_attack_state:
        state = torch.load(state_path, map_location=device)
        expected = {
            "input_size": tuple(input_size),
            "target_shape": tuple(target_s.shape),
            "steal_model": steal_model,
            **(attack_metadata or {}),
        }
        actual = state.get("metadata", {})
        mismatches = [
            f"{key}: expected {value!r}, found {actual.get(key)!r}"
            for key, value in expected.items()
            if actual.get(key) != value
        ]
        if mismatches:
            mismatch_text = "; ".join(mismatches)
            raise ValueError(
                f"Attack state {state_path} does not match this run ({mismatch_text}). "
                "Use --reset-attack-state or a different --attack-state path."
            )

        clone_client.load_state_dict(state["clone_client_state_dict"])
        with torch.no_grad():
            x_hat.copy_(state["x_hat"].to(device))
            if clamp:
                x_hat.clamp_(0.0, 1.0)
        completed_iters = int(state.get("completed_iters", 0))
        input_opt_state = state.get("input_optimizer_state_dict")
        model_opt_state = state.get("model_optimizer_state_dict")
        print(f"Loaded attack state: {state_path} (completed_iters={completed_iters})")
    elif state_path is not None and reset_attack_state and state_path.exists():
        print(f"Resetting attack state; ignoring existing state: {state_path}")

    input_opt = torch.optim.Adam([x_hat], lr=lr_input, amsgrad=True)
    if input_opt_state is not None:
        input_opt.load_state_dict(input_opt_state)

    model_opt: Optional[torch.optim.Optimizer] = None
    if steal_model:
        model_opt = torch.optim.Adam(clone_client.parameters(), lr=lr_model, amsgrad=True)
        if model_opt_state is not None:
            model_opt.load_state_dict(model_opt_state)

    mse = nn.MSELoss()
    target_s = target_s.detach()
    previous_x_hat = x_hat.detach().clone()
    stable_main_steps = 0

    def save_attack_state(total_iters: int) -> None:
        if state_path is None:
            return
        state_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "input_size": tuple(input_size),
            "target_shape": tuple(target_s.shape),
            "steal_model": steal_model,
            **(attack_metadata or {}),
        }
        torch.save(
            {
                "completed_iters": total_iters,
                "x_hat": x_hat.detach().cpu(),
                "clone_client_state_dict": clone_client.state_dict(),
                "input_optimizer_state_dict": input_opt.state_dict(),
                "model_optimizer_state_dict": model_opt.state_dict() if model_opt is not None else None,
                "metadata": metadata,
            },
            state_path,
        )
        print(f"Saved attack state: {state_path} (completed_iters={total_iters})")

    total_target_iters = completed_iters + main_iters
    print(
        f"Running attack for {main_iters} additional main iterations "
        f"({completed_iters} -> {total_target_iters})."
    )

    for local_iter in range(main_iters):
        total_iter = completed_iters + local_iter + 1
        clone_client.eval()
        for _ in range(input_iters):
            input_opt.zero_grad(set_to_none=True)
            model_input = input_transform(x_hat) if input_transform is not None else x_hat
            pred_s = clone_client(model_input)
            loss = mse(pred_s, target_s) + lambda_tv * TV(x_hat) + lambda_l2 * l2loss(x_hat)
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
                model_loss = mse(pred_s, target_s)
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

        should_log = log_every and (total_iter % log_every == 0 or local_iter == 0)
        if should_log:
            print(
                f"iter {total_iter:04d}/{total_target_iters} "
                f"(run +{local_iter + 1}/{main_iters}) | "
                f"smashed_mse={match_loss:.6f} | "
                f"mean_abs_input_change={input_change:.8f} | "
                f"stable_main_steps={stable_main_steps}/{main_convergence_patience}"
            )

        if state_save_every and (total_iter % state_save_every == 0):
            save_attack_state(total_iter)

        if (
            not disable_input_convergence
            and local_iter + 1 >= min_main_iters
            and stable_main_steps >= main_convergence_patience
        ):
            print(
                f"Converged at iter {total_iter:04d}/{total_target_iters}: "
                f"smashed_mse={match_loss:.6f}, "
                f"mean_abs_input_change={input_change:.8f}, "
                f"stable_main_steps={stable_main_steps}/{main_convergence_patience}, "
                f"input_change_tol={input_change_tol:.2e}"
            )
            save_attack_state(total_iter)
            return x_hat.detach(), total_iter

    save_attack_state(total_target_iters)
    return x_hat.detach(), total_target_iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/cifar")
    parser.add_argument("--victim-model", choices=("mobilenetv1", "cifarnet"), default="mobilenetv1",
                        help="Victim architecture to train/load and attack.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--split-children", type=int, default=3,
                        help="Number of MobileNetV1 feature children used as the client model.")
    parser.add_argument("--split-layer", type=int, default=1,
                        help="CifarNet split layer index using the original unsplit numbering (0-17).")
    parser.add_argument("--main-iters", type=int, default=1000)
    parser.add_argument("--input-iters", type=int, default=100)
    parser.add_argument("--model-iters", type=int, default=100)
    parser.add_argument("--lambda-tv", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--input-change-tol", "--input-loss-tol", dest="input_change_tol", type=float, default=1e-7,
                        help="Stop early when mean absolute reconstructed-input change stays below this tolerance.")
    parser.add_argument("--main-convergence-patience", "--convergence-patience",
                        dest="main_convergence_patience", type=int, default=5,
                        help="Number of consecutive stable main iterations required for convergence.")
    parser.add_argument("--min-main-iters", type=int, default=50,
                        help="Minimum additional main iterations before convergence can stop the attack.")
    parser.add_argument("--disable-input-convergence", action="store_true",
                        help="Ignore reconstructed-input convergence and run until main-iters is reached.")
    parser.add_argument("--known-client", action="store_true",
                        help="Use a copy of the victim client instead of stealing a random clone.")
    parser.add_argument("--save-dir", type=str, default="",
                        help="Directory for recovered/target images. Defaults to a model-specific results folder.")
    parser.add_argument("--attack-state", type=str, default="",
                        help="Path for resumable attack state. Defaults to <save-dir>/attack_state.pt.")
    parser.add_argument("--reset-attack-state", action="store_true",
                        help="Ignore an existing attack state and start the attack from scratch.")
    parser.add_argument("--state-save-every", type=int, default=50,
                        help="Save attack state every N cumulative main iterations. Use 0 to save only at the end.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint for a trained victim model.")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR),
                        help="Directory to search for or save trained victim weights.")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Stop before running if CUDA is not available.")
    args = parser.parse_args()
    if not args.save_dir:
        args.save_dir = f"results_{args.victim_model}_attack"
    save_dir = Path(args.save_dir)
    attack_state_path = Path(args.attack_state) if args.attack_state else save_dir / DEFAULT_ATTACK_STATE_NAME

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("ERROR: CUDA is required for this run, but torch.cuda.is_available() is false.")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    normalized_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    model_transform = normalized_transform if args.victim_model == "mobilenetv1" else transform
    input_transform = cifar10_normalize_tensor if args.victim_model == "mobilenetv1" else None

    trainset = datasets.CIFAR10(args.data_root, download=True, train=True, transform=transform)
    testset = datasets.CIFAR10(args.data_root, download=True, train=False, transform=transform)
    model_trainset = datasets.CIFAR10(args.data_root, download=True, train=True, transform=model_transform)
    model_testset = datasets.CIFAR10(args.data_root, download=True, train=False, transform=model_transform)

    model_name = format_model_name(args.victim_model)
    checkpoint_name = MODEL_CHECKPOINT_NAMES[args.victim_model]

    victim = build_victim_model(args.victim_model, args.num_classes).to(device)

    if args.checkpoint:
        load_model_checkpoint(victim, Path(args.checkpoint), device)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        existing_checkpoint = find_existing_checkpoint_for_model(checkpoint_dir, checkpoint_name)
        if existing_checkpoint is not None:
            load_model_checkpoint(victim, existing_checkpoint, device)
        else:
            print(f"No {model_name} weights found in {checkpoint_dir}; training victim model...")
            test_acc = train_classifier_model(
                victim,
                model_trainset,
                model_testset,
                device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            save_model_checkpoint(victim, checkpoint_dir / checkpoint_name, args.epochs, test_acc)

    victim.eval()
    victim_acc = accuracy(victim, model_testset, device, batch_size=args.batch_size)
    print(f"Victim model test accuracy before attack: {victim_acc:.2f}%")
    if victim_acc < MIN_ATTACK_MODEL_ACCURACY:
        raise SystemExit(
            "ERROR: Victim model accuracy is below the attack sanity threshold: "
            f"{victim_acc:.2f}% < {MIN_ATTACK_MODEL_ACCURACY:.2f}%. "
            "Stopping before attack."
        )

    images = torch.stack(
        [get_examples_by_class(testset, c, count=1) for c in range(args.num_classes)],
        dim=0,
    ).to(device)
    true_labels = torch.arange(args.num_classes, device=device)

    victim_client = build_client_model(
        victim,
        args.victim_model,
        split_children=args.split_children,
        split_layer=args.split_layer,
    ).to(device)
    victim_client.eval()

    with torch.no_grad():
        model_input = input_transform(images) if input_transform is not None else images
        target_s = victim_client(model_input)

    print(f"Victim model: {model_name}")
    if args.victim_model == "mobilenetv1":
        print(f"Client split children: {args.split_children}")
    else:
        print(f"Client split layer: {args.split_layer}")
    print(f"Target full smashed representation shape: {tuple(target_s.shape)}")
    print(f"Attack regularization: lambda_tv={args.lambda_tv:g}, lambda_l2={args.lambda_l2:g}")

    if args.known_client:
        clone_client = copy.deepcopy(victim_client).to(device)
        steal_model = False
        print("Attack mode: known client; optimizing only x_hat.")
    else:
        random_victim = build_victim_model(args.victim_model, args.num_classes).to(device)
        clone_client = build_client_model(
            random_victim,
            args.victim_model,
            split_children=args.split_children,
            split_layer=args.split_layer,
        ).to(device)
        steal_model = True
        print("Attack mode: random clone client; optimizing both x_hat and clone_client.")

    result, total_attack_iters = full_smashed_inversion_attack(
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
        input_transform=input_transform,
        steal_model=steal_model,
        state_path=attack_state_path,
        reset_attack_state=args.reset_attack_state,
        state_save_every=args.state_save_every,
        attack_metadata={
            "victim_model": args.victim_model,
            "num_classes": args.num_classes,
            "split_children": args.split_children if args.victim_model == "mobilenetv1" else None,
            "split_layer": args.split_layer if args.victim_model == "cifarnet" else None,
            "known_client": args.known_client,
            "lambda_tv": args.lambda_tv,
            "lambda_l2": args.lambda_l2,
            "input_iters": args.input_iters,
            "model_iters": args.model_iters,
            "input_change_tol": args.input_change_tol,
            "main_convergence_patience": args.main_convergence_patience,
            "min_main_iters": args.min_main_iters,
            "input_transform": "cifar10_normalize" if input_transform is not None else None,
        },
    )

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
    print(f"Total cumulative attack iterations: {total_attack_iters}")
    restored_acc, restored_correct, restored_total = restored_input_accuracy(
        victim,
        result,
        true_labels,
        device,
        input_transform=input_transform,
    )
    print(
        "Restored-input clone accuracy (victim model on restored inputs): "
        f"{restored_acc:.2f}% ({restored_correct}/{restored_total})"
    )
    print(f"Saved recovered and target images to: {save_dir}")
    print(f"Saved attack state to: {attack_state_path}")


if __name__ == "__main__":
    main()
