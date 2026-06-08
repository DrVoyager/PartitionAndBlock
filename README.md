# Partition and Blocking CIFAR-10 Experiments

This repository contains PyTorch code for evaluating Partition and Blocking
(P&B) split-inference defenses and UnSplit-style reconstruction attacks on
CIFAR-10.

## What is Included

- `mobilenet_v1.py`: P&B model definition. The client produces smashed features
  with shape `(64, 16, 16)`. The central feature partition is processed by the
  protected branch, while the full smashed tensor is also processed by the
  original branch.
- `partition_full_smashed_attack.py`: Full-smashed reconstruction attack against
  the P&B victim model.
- `partition_blocked_smashed_attack.py`: Blocked-smashed reconstruction attack
  against the P&B victim model. The attack loss only matches the noncentral
  smashed features.
- `unsplit_attack_on_full_smashed_layer.py`: Full-smashed attack runner for
  standard MobileNet V1 and CifarNet victims. Supports known-client and
  unknown-client attack modes.
- `models.py`: CifarNet implementation used by the UnSplit-style attack runner.
- `train.py`, `train_baseline.py`, `evaluate.py`, `compare_models.py`: Training,
  evaluation, and model comparison utilities.
- `util.py`: Shared regularization, normalization, and CIFAR-10 sample helpers.

Generated outputs such as checkpoints, CIFAR-10 data, SLURM logs, and result
images are intentionally ignored by Git.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The scripts download CIFAR-10 automatically when needed. By default, data is
stored under `data/cifar` or `data` depending on the script.

## Training and Evaluation

Train the P&B model:

```bash
python train.py
```

Train the baseline MobileNet V1 model:

```bash
python train_baseline.py
```

Evaluate the trained P&B model:

```bash
python evaluate.py
```

Compare the trained P&B and baseline models:

```bash
python compare_models.py
```

Expected checkpoint locations:

```text
checkpoints/best_model.pth
checkpoints/baseline_mobilenetv1.pth
```

## Reconstruction Attacks

### P&B Full-Smashed Attack

Known-client:

```bash
python partition_full_smashed_attack.py \
  --checkpoint checkpoints/partition_model_attack_victim.pth \
  --known-client \
  --main-iters 1000 \
  --input-iters 100 \
  --model-iters 100 \
  --save-dir results_partition_full_known
```

Unknown-client:

```bash
python partition_full_smashed_attack.py \
  --checkpoint checkpoints/partition_model_attack_victim.pth \
  --main-iters 1000 \
  --input-iters 100 \
  --model-iters 100 \
  --save-dir results_partition_full_unknown
```

### P&B Blocked-Smashed Attack

Known-client:

```bash
python partition_blocked_smashed_attack.py \
  --checkpoint checkpoints/partition_model_attack_victim.pth \
  --known-client \
  --main-iters 1000 \
  --input-iters 100 \
  --model-iters 100 \
  --save-dir results_partition_blocked_known
```

Unknown-client:

```bash
python partition_blocked_smashed_attack.py \
  --checkpoint checkpoints/partition_model_attack_victim.pth \
  --main-iters 1000 \
  --input-iters 100 \
  --model-iters 100 \
  --save-dir results_partition_blocked_unknown
```

The blocked attack masks smashed feature rows `5:10` and columns `5:10` across
all 64 channels. Pixel MSE is still computed over the full recovered image.

### MobileNet V1 / CifarNet Full-Smashed Attack

MobileNet V1 unknown-client attack:

```bash
python unsplit_attack_on_full_smashed_layer.py \
  --victim-model mobilenetv1 \
  --main-iters 1000 \
  --input-iters 100 \
  --model-iters 100 \
  --checkpoint-dir checkpoints_mobilenetv1 \
  --save-dir results_mobilenetv1_unknown \
  --reset-attack-state
```

CifarNet unknown-client attack:

```bash
python unsplit_attack_on_full_smashed_layer.py \
  --victim-model cifarnet \
  --split-layer 3 \
  --main-iters 1000 \
  --input-iters 100 \
  --model-iters 100 \
  --checkpoint-dir checkpoints_cifarnet \
  --save-dir results_cifarnet_unknown_split3 \
  --reset-attack-state
```

Add `--known-client` to either command to run the stronger known-client attack.

## Outputs

Attack scripts save:

```text
target_0.png ... target_9.png
recovered_0.png ... recovered_9.png
```

Some scripts also save `attack_state.pt` so unknown-client attacks can be
resumed. Use `--reset-attack-state` to force a fresh attack run.

## Notes

- Unknown-client attacks are stochastic because the clone client is randomly
  initialized unless a seed is explicitly added.
- Clone accuracy is reported as the victim model's classification accuracy on
  the restored inputs.
- Result directories, checkpoints, data, and logs are excluded by `.gitignore`.
