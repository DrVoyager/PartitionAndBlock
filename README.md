# Partition and Block CIFAR-10 Experiments

This repository contains PyTorch code for evaluating Partition and Block
(P&B) split-inference defenses and UnSplit-style reconstruction attacks on
CIFAR-10.

P&B partitions smashed data and hides a centered spatial block from the
untrusted server path. The protected block is processed by protected server
layers, while the remaining smashed representation is processed by the ordinary
server path and later merged for classification.

## What is Included

- `mobilenet_v1.py`: P&B MobileNetV1 model definition. The default split is the
  MobileNetV1 stem, which produces smashed features with shape `(32, 32, 32)`.
  The model also supports configurable split points after MobileNetV1 depthwise
  blocks and configurable centered block size.
- `train.py`: trains a P&B MobileNetV1 victim model. Supports split and block
  configuration through CLI flags.
- `train_baseline.py`: trains the baseline MobileNetV1 model.
- `mobilenetv1_full_smashed_attack.py`: baseline MobileNetV1 full-smashed
  UnSplit-style attack runner. Supports known-client and unknown-client modes.
- `partition_blocked_smashed_attack.py`: P&B blocked-smashed attack runner. The
  attacker only matches observable, nonprotected smashed features.
- `partition_full_smashed_attack.py`: legacy full-smashed attack against the
  P&B victim model.
- `unsplit_attack_on_full_smashed_layer.py`: general full-smashed attack runner
  for MobileNetV1 and CifarNet victims.
- `cifarnet_full_smashed_attack.py`: CifarNet full-smashed split-depth attack
  runner.
- `cifarnet_partition_blocked_smashed_attack.py`: CifarNet P&B blocked-smashed
  split-depth attack runner.
- `models.py`: CifarNet implementation used by CifarNet attack scripts.
- `evaluate.py`, `compare_models.py`: model evaluation and comparison helpers.
- `util.py`: shared regularization, normalization, and CIFAR-10 sample helpers.

Generated outputs such as checkpoints, CIFAR-10 data, SLURM logs, and result
images are intentionally ignored by Git.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The scripts download CIFAR-10 automatically when needed. Most scripts default
to `data/cifar`; `train.py` defaults to `./data`.

## MobileNetV1 Split and Block Options

The current MobileNetV1 scripts support these split options:

- `--split-preset stem`: split before the first depthwise block. This is the
  default and uses the MobileNetV1 stem output.
- `--split-preset dw1` ... `--split-preset dw12`: split after the selected
  depthwise block.
- `--split-after-depthwise N`: explicit equivalent of `dwN`.
- `--split-children N`: low-level override for the number of MobileNetV1
  feature children in the client.

For P&B MobileNetV1, `--mask-side M` sets the side length of the centered
protected block. If omitted, the model uses the split-dependent default
`height // 3`.

The protected block is encoded by four standard convolutions followed by
adaptive average pooling and a 256-dimensional linear projection. Two
independent options control this encoder:

- `--protected-width R` scales the convolutional channels from the base widths
  `(32, 64, 128)` and rounds each result to a multiple of eight.
- `--protected-pool-size P` retains a fixed `P x P` spatial representation
  before the linear projection.

The defaults are `R=1.0` and `P=2`. At the default stem split, the protected
encoder receives `(B, 32, M, M)` and contains 380,864 trainable parameters.
Its parameter count is independent of `M`. Existing checkpoints created with
the previous depthwise/pointwise protected branch are not architecture
compatible and must be retrained on this branch.

Default stem/block-size-10 behavior:

```text
client output:       (B, 32, 32, 32)
protected block:     rows 11:21, columns 11:21, all 32 channels
observed ratio:      0.9023
protected ratio:     0.0977
```

For new split points or new `--mask-side` values, retrain the P&B victim model
and use the matching checkpoint for attacks.

## Training

Train the default stem-split P&B MobileNetV1 model:

```bash
python train.py \
  --protected-width 1.0 \
  --protected-pool-size 2 \
  --epochs 35 \
  --checkpoint-dir checkpoints \
  --checkpoint-name best_model.pth
```

Train a P&B MobileNetV1 model with a larger protected block:

```bash
python train.py \
  --mask-side 16 \
  --epochs 35 \
  --checkpoint-dir checkpoints \
  --checkpoint-name best_model_mask16.pth
```

Train a P&B MobileNetV1 model split after the first depthwise block:

```bash
python train.py \
  --split-after-depthwise 1 \
  --epochs 35 \
  --checkpoint-dir checkpoints \
  --checkpoint-name best_model_dw1.pth
```

Train the baseline MobileNetV1 model:

```bash
python train_baseline.py
```

Expected default checkpoint locations:

```text
checkpoints/best_model.pth
checkpoints/baseline_mobilenetv1.pth
```

## Evaluation

Evaluate the trained default P&B model:

```bash
python evaluate.py
```

Compare the trained P&B and baseline models:

```bash
python compare_models.py
```

## MobileNetV1 Reconstruction Attacks

### Baseline MobileNetV1 Full-Smashed Attack

Known-client stem split:

```bash
python mobilenetv1_full_smashed_attack.py \
  --checkpoint checkpoints/baseline_mobilenetv1.pth \
  --split-preset stem \
  --known-client \
  --main-iters 10000 \
  --input-iters 100 \
  --model-iters 100 \
  --input-change-tol 1e-4 \
  --save-dir results_mobilenetv1_full_known
```

Unknown-client stem split:

```bash
python mobilenetv1_full_smashed_attack.py \
  --checkpoint checkpoints/baseline_mobilenetv1.pth \
  --split-preset stem \
  --main-iters 10000 \
  --input-iters 100 \
  --model-iters 100 \
  --input-change-tol 1e-4 \
  --save-dir results_mobilenetv1_full_unknown
```

To attack a later split, add for example:

```bash
--split-after-depthwise 1
```

### P&B Blocked-Smashed Attack

Known-client default stem/block-size-10 setting:

```bash
python partition_blocked_smashed_attack.py \
  --checkpoint checkpoints/best_model.pth \
  --split-preset stem \
  --known-client \
  --main-iters 10000 \
  --input-iters 100 \
  --model-iters 100 \
  --input-change-tol 1e-4 \
  --main-convergence-patience 5 \
  --min-main-iters 50 \
  --lambda-tv 0.1 \
  --lambda-l2 0 \
  --save-dir results_partition_blocked_known
```

Unknown-client default stem/block-size-10 setting:

```bash
python partition_blocked_smashed_attack.py \
  --checkpoint checkpoints/best_model.pth \
  --split-preset stem \
  --main-iters 10000 \
  --input-iters 100 \
  --model-iters 100 \
  --input-change-tol 1e-4 \
  --main-convergence-patience 5 \
  --min-main-iters 50 \
  --lambda-tv 0.1 \
  --lambda-l2 0 \
  --save-dir results_partition_blocked_unknown
```

For a model trained with `--mask-side 16`, attack with the same value:

```bash
python partition_blocked_smashed_attack.py \
  --checkpoint checkpoints/best_model_mask16.pth \
  --mask-side 16 \
  --known-client \
  --main-iters 10000 \
  --input-change-tol 1e-4 \
  --save-dir results_partition_blocked_mask16_known
```

The blocked-smashed attack reports the full smashed shape, protected block
bounds, masked feature ratio, observed feature ratio, average pixel MSE, and
restored-input clone accuracy. Use `--disable-input-convergence` to force the
attack to run until `--main-iters`.

## CifarNet Reconstruction Attacks

CifarNet scripts support split depths `1` through `6`, following the UnSplit
CifarNet split-depth convention.

Full-smashed unknown-client attack:

```bash
python cifarnet_full_smashed_attack.py \
  --split-layer 1 \
  --main-iters 10000 \
  --input-iters 100 \
  --model-iters 100 \
  --input-change-tol 1e-4 \
  --save-dir results_cifarnet_full_unknown_split1
```

Full-smashed known-client attack:

```bash
python cifarnet_full_smashed_attack.py \
  --split-layer 1 \
  --known-client \
  --main-iters 10000 \
  --input-iters 100 \
  --model-iters 100 \
  --input-change-tol 1e-4 \
  --save-dir results_cifarnet_full_known_split1
```

P&B blocked-smashed unknown-client attack:

```bash
python cifarnet_partition_blocked_smashed_attack.py \
  --split-layer 1 \
  --main-iters 10000 \
  --input-iters 100 \
  --model-iters 100 \
  --input-change-tol 1e-4 \
  --save-dir results_cifarnet_pb_unknown_split1
```

P&B blocked-smashed known-client attack:

```bash
python cifarnet_partition_blocked_smashed_attack.py \
  --split-layer 1 \
  --known-client \
  --main-iters 10000 \
  --input-iters 100 \
  --model-iters 100 \
  --input-change-tol 1e-4 \
  --save-dir results_cifarnet_pb_known_split1
```

For CifarNet P&B, `--mask-side` optionally overrides the centered protected
block size. If omitted, the script blocks the centered spatial block for the
current smashed tensor.

## General UnSplit Runner

`unsplit_attack_on_full_smashed_layer.py` remains available as a general runner
for MobileNetV1 and CifarNet:

```bash
python unsplit_attack_on_full_smashed_layer.py \
  --victim-model mobilenetv1 \
  --split-children 3 \
  --main-iters 1000 \
  --input-iters 100 \
  --model-iters 100 \
  --checkpoint-dir checkpoints_mobilenetv1 \
  --save-dir results_mobilenetv1_unknown \
  --reset-attack-state
```

For CifarNet:

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

Add `--known-client` to run the stronger known-client attack.

## Attack Objective and Convergence

The current attack objective defaults to feature MSE plus Total Variation:

```text
--lambda-tv 0.1
--lambda-l2 0.0
```

Unknown-client attacks optimize both the recovered inputs and a clone client.
Known-client attacks copy the victim client and optimize only the recovered
inputs.

Attack loops can stop when reconstructed input change converges:

```text
--input-change-tol 1e-7
--main-convergence-patience 5
--min-main-iters 50
```

Use `--disable-input-convergence` to ignore convergence and run until
`--main-iters`.

## Outputs

Attack scripts save image artifacts such as:

```text
target_0.png ... target_9.png
recovered_0.png ... recovered_9.png
```

Some scripts also save `attack_state.pt` so unknown-client attacks can be
resumed. Use `--reset-attack-state` to force a fresh attack run when supported.

## Notes

- Use `--require-cuda` on HPC runs when CPU fallback would be too slow.
- Unknown-client attacks are stochastic because the clone client is randomly
  initialized unless a seed is fixed.
- Restore-input accuracy is reported as the victim model's classification
  accuracy on the recovered inputs.
- New P&B split points and new block sizes require matching checkpoints.
- Result directories, checkpoints, data, and logs are excluded by `.gitignore`.
