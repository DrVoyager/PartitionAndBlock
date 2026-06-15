# New MobileNetV1 and P&B MobileNet Training Report

**Date:** June 13, 2026
**Cluster:** `hpc.cpp.edu`
**Conda environment:** `/data03/home/yongzhiwang/anaconda/envs/unsplit/`
**Slurm job:** `41172`, array tasks `41172_0` and `41172_1`
**Local run directory:** `partition_and_blocking_20260604/training_runs/pb_mobilenet_corrected_20260613_094951/`
**Remote run directory:** `/data03/home/yongzhiwang/develop/partition_and_blocking_20260604/training_runs/pb_mobilenet_corrected_20260613_094951/`

## 1. Objective

This run retrained both models after correcting the Partition and Blocking
MobileNet implementation so that its client-side split matches the standard
MobileNetV1 stem:

```text
Conv2d(3 -> 32, 3x3, stride=1, padding=1) -> BatchNorm2d(32) -> ReLU
```

The prior P&B client included an extra convolutional layer and produced
`[B, 64, 16, 16]` smashed data. The corrected P&B client now produces
`[B, 32, 32, 32]` smashed data, matching the output of the MobileNetV1 stem.

This report records the corrected architectures, training settings, Slurm job
details, checkpoint locations, and training results.

## 2. Source Files

The run used the following source files uploaded to the timestamped HPC run
directory:

| Purpose                              | File                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------- |
| Corrected P&B model definition       | `partition_and_blocking_20260604/code/partition_and_block_mobilenet_model.py` |
| Corrected P&B training script        | `partition_and_blocking_20260604/code/train.py`                               |
| Baseline MobileNetV1 training script | `partition_and_blocking_20260604/code/train_baseline.py`                      |

The training run directory also symlinked the existing remote CIFAR-10 data
directory:

```text
/data03/home/yongzhiwang/develop/partition_and_blocking_20260604/data
```

## 3. Corrected P&B Architecture

### 3.1 Client Model

The corrected P&B client is the MobileNetV1 stem only:

```text
Input: [B, 3, 32, 32]
Conv2d(3 -> 32, kernel=3, stride=1, padding=1, bias=False)
BatchNorm2d(32)
ReLU
Output smashed data: [B, 32, 32, 32]
```

### 3.2 Protected Server Layers

The P&B model computes the protected partition size from the client output:

```text
H = 32, W = 32
grid_h = H // 3 = 10
grid_w = W // 3 = 10
```

The central protected partition is extracted with:

```text
h_start = (32 - 10 + 1) // 2 = 11
h_end   = 21
w_start = (32 - 10 + 1) // 2 = 11
w_end   = 21
```

Thus, the protected smashed partition is:

```text
Protected partition: [B, 32, 10, 10]
Flatten: 3200
Linear(3200 -> 512)
ReLU
Dropout(0.3)
Linear(512 -> 256)
ReLU
Protected feature z_P: [B, 256]
```

The protected partition contains `32 * 10 * 10 = 3200` smashed-feature values
out of `32 * 32 * 32 = 32768`, so the blocked portion is `9.766%` of the
smashed representation and the untrusted server-visible portion is `90.234%`.

### 3.3 Original Server Layers

The original server branch receives the full client smashed tensor and applies
the remaining MobileNetV1 depthwise separable blocks after the stem:

```text
Input: [B, 32, 32, 32]
Depthwise block 32 -> 64, stride=1
Depthwise block 64 -> 128, stride=2
Depthwise block 128 -> 128, stride=1
Depthwise block 128 -> 256, stride=2
Depthwise block 256 -> 256, stride=1
Depthwise block 256 -> 512, stride=2
Depthwise block 512 -> 512, stride=1
Depthwise block 512 -> 512, stride=1
Depthwise block 512 -> 512, stride=1
Depthwise block 512 -> 512, stride=1
Depthwise block 512 -> 1024, stride=2
Depthwise block 1024 -> 1024, stride=1
AdaptiveAvgPool2d(1)
Flatten
Linear(1024 -> 256)
Original feature z_O: [B, 256]
```

Each depthwise block is:

```text
Depthwise Conv2d(in_ch -> in_ch, kernel=3, groups=in_ch)
BatchNorm2d(in_ch)
ReLU
Pointwise Conv2d(in_ch -> out_ch, kernel=1)
BatchNorm2d(out_ch)
ReLU
```

### 3.4 Merging Layers

The merging network is unchanged from the prior P&B design:

```text
Concat(z_P, z_O): [B, 512]
Linear(512 -> 512)
ReLU
Dropout(0.3)
Linear(512 -> 256)
ReLU
Linear(256 -> 10)
Output logits: [B, 10]
```

### 3.5 P&B Parameter Count and Shape Check

Local verification after training-file synchronization:

| Quantity                  |               Value |
| ------------------------- | ------------------: |
| Total parameters          |           5,367,370 |
| Trainable parameters      |           5,367,370 |
| Client smashed shape      | `(2, 32, 32, 32)` |
| Protected partition shape | `(2, 32, 10, 10)` |
| Protected branch output   |        `(2, 256)` |
| Original branch output    |        `(2, 256)` |
| Final output              |         `(2, 10)` |

## 4. Baseline MobileNetV1 Architecture

The baseline is a standard CIFAR-10 MobileNetV1-style model:

```text
Input: [B, 3, 32, 32]
Conv2d(3 -> 32, kernel=3, stride=1, padding=1, bias=False)
BatchNorm2d(32)
ReLU
Depthwise block 32 -> 64, stride=1
Depthwise block 64 -> 128, stride=2
Depthwise block 128 -> 128, stride=1
Depthwise block 128 -> 256, stride=2
Depthwise block 256 -> 256, stride=1
Depthwise block 256 -> 512, stride=2
Depthwise block 512 -> 512, stride=1
Depthwise block 512 -> 512, stride=1
Depthwise block 512 -> 512, stride=1
Depthwise block 512 -> 512, stride=1
Depthwise block 512 -> 1024, stride=2
Depthwise block 1024 -> 1024, stride=1
AdaptiveAvgPool2d(1)
Flatten
Linear(1024 -> 10)
Output logits: [B, 10]
```

Parameter count:

| Quantity             |     Value |
| -------------------- | --------: |
| Total parameters     | 2,948,426 |
| Trainable parameters | 2,948,426 |

## 5. Training Configuration

Both models were trained on CIFAR-10 with the same data preprocessing and
optimizer settings.

### Dataset and Data Augmentation

Training transform:

```text
RandomCrop(32, padding=4)
RandomHorizontalFlip()
ToTensor()
Normalize(mean=(0.4914, 0.4822, 0.4465),
          std=(0.2023, 0.1994, 0.2010))
```

Test transform:

```text
ToTensor()
Normalize(mean=(0.4914, 0.4822, 0.4465),
          std=(0.2023, 0.1994, 0.2010))
```

Common loader settings:

| Setting            |    Value |
| ------------------ | -------: |
| Dataset            | CIFAR-10 |
| Train batch size   |      128 |
| Test batch size    |      128 |
| DataLoader workers |        2 |
| Device             |     CUDA |

### Optimizer

| Setting            |               P&B |         Baseline |
| ------------------ | ----------------: | ---------------: |
| Loss               |  CrossEntropyLoss | CrossEntropyLoss |
| Optimizer          |              Adam |             Adam |
| Learning rate      |             0.001 |            0.001 |
| Weight decay       |              5e-4 |             5e-4 |
| Scheduler          | ReduceLROnPlateau |             None |
| Scheduler mode     |               max |              N/A |
| Scheduler factor   |               0.5 |              N/A |
| Scheduler patience |                 5 |              N/A |
| Epochs             |                35 |               76 |

## 6. Slurm Job Details

The models were trained through Slurm array job `41172`.

|  Array task | Model                   | Script                     | Status    |
| ----------: | ----------------------- | -------------------------- | --------- |
| `41172_0` | Corrected P&B MobileNet | `code/train.py`          | Completed |
| `41172_1` | Baseline MobileNetV1    | `code/train_baseline.py` | Completed |

The job requested:

```text
partition: gpu
gpus: 1
cpus-per-task: 4
memory: 32G
time limit: 24:00:00
```

The Slurm script was saved remotely as:

```text
/data03/home/yongzhiwang/develop/partition_and_blocking_20260604/training_runs/pb_mobilenet_corrected_20260613_094951/train_both.sbatch
```

## 7. Training Results

### 7.1 Corrected P&B MobileNet

The corrected P&B model was trained for 35 epochs. The best checkpoint was
saved whenever validation accuracy improved.

| Metric                    |  Value |
| ------------------------- | -----: |
| Final epoch               |     35 |
| Final train loss          | 0.3631 |
| Final train accuracy      | 87.92% |
| Final validation loss     | 0.4348 |
| Final validation accuracy | 86.23% |
| Best validation accuracy  | 86.23% |
| Best checkpoint epoch     |     35 |

Selected validation trajectory:

| Epoch | Validation Accuracy |
| ----: | ------------------: |
|     1 |              43.92% |
|     5 |              71.45% |
|    10 |              78.65% |
|    15 |              82.89% |
|    20 |              83.98% |
|    25 |              84.77% |
|    30 |              84.99% |
|    32 |              86.13% |
|    35 |              86.23% |

### 7.2 Baseline MobileNetV1

The baseline MobileNetV1 model was trained for 76 epochs. The current
`train_baseline.py` script tracks `best_acc` internally, but saves the final
epoch checkpoint rather than the best-validation checkpoint.

| Metric                            |  Value |
| --------------------------------- | -----: |
| Final epoch                       |     76 |
| Final train accuracy              | 91.56% |
| Final validation accuracy         | 86.93% |
| Best observed validation accuracy | 88.46% |
| Best observed epoch               |     60 |
| Saved checkpoint epoch            |     76 |

Selected validation trajectory:

| Epoch | Validation Accuracy |
| ----: | ------------------: |
|     1 |              43.26% |
|     5 |              69.40% |
|    10 |              78.71% |
|    20 |              82.66% |
|    30 |              84.85% |
|    40 |              86.43% |
|    50 |              86.62% |
|    60 |              88.46% |
|    70 |              88.34% |
|    76 |              86.93% |

### 7.3 Accuracy Comparison

| Model                   | Saved Checkpoint Accuracy | Best Observed Accuracy | Saved Epoch |
| ----------------------- | ------------------------: | ---------------------: | ----------: |
| Baseline MobileNetV1    |                    86.93% |                 88.46% |          76 |
| Corrected P&B MobileNet |                    86.23% |                 86.23% |          35 |

Compared with the saved baseline checkpoint, the corrected P&B checkpoint has a
0.70 percentage-point lower validation accuracy. Compared with the baseline's
best observed validation accuracy, the corrected P&B checkpoint is 2.23
percentage points lower.

## 8. Checkpoints and Logs

Downloaded local files:

```text
partition_and_blocking_20260604/training_runs/pb_mobilenet_corrected_20260613_094951/
├── checkpoints/
│   ├── best_model.pth
│   └── baseline_mobilenetv1.pth
└── slurm_logs/
    ├── pb_mobilenet_train_41172_0.out
    ├── pb_mobilenet_train_41172_0.err
    ├── pb_mobilenet_train_41172_1.out
    └── pb_mobilenet_train_41172_1.err
```

Checkpoint metadata:

| Checkpoint                   |             Size | Keys                                                                    | Accuracy Field       |
| ---------------------------- | ---------------: | ----------------------------------------------------------------------- | -------------------- |
| `best_model.pth`           | 64,620,645 bytes | `epoch`, `model_state_dict`, `optimizer_state_dict`, `best_acc` | `best_acc = 86.23` |
| `baseline_mobilenetv1.pth` | 11,925,678 bytes | `epoch`, `model_state_dict`, `val_acc`                            | `val_acc = 86.93`  |

## 9. Notes for Future Experiments

1. The old P&B attack results were produced with the previous P&B architecture
   whose client output was `[B, 64, 16, 16]`. Those attack metrics are not
   directly comparable to the corrected P&B model.
2. Future UnSplit-style attacks should use the corrected checkpoint:

```text
partition_and_blocking_20260604/training_runs/pb_mobilenet_corrected_20260613_094951/checkpoints/best_model.pth
```

3. Baseline MobileNetV1 attack reruns should use:

```text
partition_and_blocking_20260604/training_runs/pb_mobilenet_corrected_20260613_094951/checkpoints/baseline_mobilenetv1.pth
```

4. If the baseline training script should preserve the best validation model,
   update `train_baseline.py` to save a checkpoint whenever validation accuracy
   improves. The current baseline checkpoint is the final epoch-76 model, not
   the epoch-60 best observed model.

## 10. Corrected Stem Attack Rerun

**Date:** 2026-06-14  
**SLURM job:** `41186`  
**Remote run directory:** `/data03/home/yongzhiwang/develop/partition_and_blocking_20260604/attack_runs/corrected_stem_converge_20260614_085342/`  
**Local output directory:** `partition_and_blocking_20260604/attack_runs/corrected_stem_converge_20260614_085342/`

This rerun attacks the newly trained baseline MobileNetV1 and corrected P&B
MobileNet checkpoints. The baseline full-smashed attack uses
`mobilenetv1_full_smashed_attack.py` with `--split-children 3`, so the observed
smashed tensor is the output of the same stem layer used by the corrected P&B
client. The corrected P&B attack uses `partition_blocked_smashed_attack.py` and
the corrected P&B checkpoint.

Common attack settings:

| Setting | Value |
| ------- | ----: |
| `main-iters` | 100000 |
| `input-iters` | 100 |
| `model-iters` | 100 |
| `input-change-tol` | 0.0001 |
| `main-convergence-patience` | 5 |
| `min-main-iters` | 50 |
| Number of target images | 10 |

Checkpoint inputs:

| Model | Checkpoint |
| ----- | ---------- |
| Baseline MobileNetV1 | `training_runs/pb_mobilenet_corrected_20260613_094951/checkpoints/baseline_mobilenetv1.pth` |
| Corrected P&B MobileNet | `training_runs/pb_mobilenet_corrected_20260613_094951/checkpoints/best_model.pth` |

Attack results:

| Attack | Client Knowledge | Device Reported | Smashed Tensor | Protected Region | Observed Ratio | Converged Iteration | Final Smashed MSE | Average Pixel MSE | Restored-Input Clone Accuracy |
| ------ | ---------------- | --------------- | -------------- | ---------------- | -------------: | ------------------: | ----------------: | ----------------: | ----------------------------: |
| Baseline MobileNetV1 full-smashed | Known | CUDA | `(10, 32, 32, 32)` | None | 1.0000 | 50 | 0.001015 | 0.244023 | 10.00% (1/10) |
| Baseline MobileNetV1 full-smashed | Unknown | CUDA | `(10, 32, 32, 32)` | None | 1.0000 | 324 | 0.003049 | 0.202883 | 10.00% (1/10) |
| Corrected P&B blocked-smashed | Known | CPU | `(10, 32, 32, 32)` | `h=11:21, w=11:21` | 0.9023 | 50 | 0.001662 | 0.242714 | 10.00% (1/10) |
| Corrected P&B blocked-smashed | Unknown | CPU | `(10, 32, 32, 32)` | `h=11:21, w=11:21` | 0.9023 | 202 | 0.002179 | 0.208743 | 10.00% (1/10) |

Per-image pixel MSE:

| Class Index | Baseline Known | Baseline Unknown | P&B Known | P&B Unknown |
| ----------: | -------------: | ---------------: | --------: | ----------: |
| 0 | 0.414446 | 0.354808 | 0.412748 | 0.359100 |
| 1 | 0.149190 | 0.127430 | 0.148292 | 0.126664 |
| 2 | 0.226860 | 0.182460 | 0.225323 | 0.196069 |
| 3 | 0.211517 | 0.170193 | 0.209630 | 0.176314 |
| 4 | 0.108917 | 0.083380 | 0.107794 | 0.081964 |
| 5 | 0.173541 | 0.136428 | 0.172630 | 0.144851 |
| 6 | 0.201243 | 0.155560 | 0.200614 | 0.169661 |
| 7 | 0.163912 | 0.137053 | 0.163200 | 0.143297 |
| 8 | 0.461864 | 0.402013 | 0.459839 | 0.406913 |
| 9 | 0.328742 | 0.279500 | 0.327072 | 0.282593 |

All four tasks stopped through the reconstructed-input convergence criterion
rather than reaching the `main-iters` upper bound. The P&B jobs reported a CUDA
initialization warning and ran on CPU, while the baseline jobs ran on CUDA. The
local output directory contains all target/recovered images and
`slurm_logs/combined_metrics.txt`.

## 11. Normalized-Input Attack Correction

**Date:** 2026-06-14  
**SLURM job:** `41210`  
**Remote run directory:** `/data03/home/yongzhiwang/develop/partition_and_blocking_20260604/attack_runs/normalized_input_converge_20260614_231242/`  
**Local output directory:** `partition_and_blocking_20260604/attack_runs/normalized_input_converge_20260614_231242/`

The attack results in Section 10 were found to be inconsistent with the model
training pipeline. Both victim checkpoints were trained and evaluated with
CIFAR-10 normalization:

```text
Normalize(mean=(0.4914, 0.4822, 0.4465),
          std=(0.2023, 0.1994, 0.2010))
```

However, the attack scripts generated smashed targets from raw `[0, 1]` pixel
tensors and also computed restored-input clone accuracy by sending raw restored
tensors directly into the victim model. This mismatch made the restored-input
clone accuracy artificially collapse toward chance. The corrected attack keeps
`x_hat` in pixel space for image saving and pixel MSE, but applies CIFAR-10
normalization before every victim/client forward pass:

```text
target_s = client(normalize(x))
pred_s   = clone_client(normalize(x_hat))
RICA     = victim(normalize(x_hat))
```

Before rerunning the attack, a local checkpoint sanity check classified the ten
selected target images correctly for P&B (`10/10`) and nearly correctly for the
baseline (`9/10`) when the same normalization was applied.

Common attack settings:

| Setting | Value |
| ------- | ----: |
| `main-iters` | 100000 |
| `input-iters` | 100 |
| `model-iters` | 100 |
| `input-change-tol` | 0.0001 |
| `main-convergence-patience` | 5 |
| `min-main-iters` | 50 |
| CUDA required | Yes |
| Number of target images | 10 |

Corrected attack results:

| Attack | Client Knowledge | Device Reported | Smashed Tensor | Protected Region | Observed Ratio | Converged Iteration | Final Smashed MSE | Average Pixel MSE | Restored-Input Clone Accuracy |
| ------ | ---------------- | --------------- | -------------- | ---------------- | -------------: | ------------------: | ----------------: | ----------------: | ----------------------------: |
| Baseline MobileNetV1 full-smashed | Known | CUDA | `(10, 32, 32, 32)` | None | 1.0000 | 50 | 0.015078 | 0.207234 | 30.00% (3/10) |
| Baseline MobileNetV1 full-smashed | Unknown | CUDA | `(10, 32, 32, 32)` | None | 1.0000 | 120 | 0.003014 | 0.095883 | 10.00% (1/10) |
| Corrected P&B blocked-smashed | Known | CUDA | `(10, 32, 32, 32)` | `h=11:21, w=11:21` | 0.9023 | 50 | 0.021954 | 0.193614 | 50.00% (5/10) |
| Corrected P&B blocked-smashed | Unknown | CUDA | `(10, 32, 32, 32)` | `h=11:21, w=11:21` | 0.9023 | 216 | 0.003400 | 0.219844 | 10.00% (1/10) |

Per-image pixel MSE:

| Class Index | Baseline Known | Baseline Unknown | P&B Known | P&B Unknown |
| ----------: | -------------: | ---------------: | --------: | ----------: |
| 0 | 0.368769 | 0.159127 | 0.342754 | 0.381471 |
| 1 | 0.109306 | 0.085666 | 0.100965 | 0.136501 |
| 2 | 0.184521 | 0.062293 | 0.178042 | 0.199017 |
| 3 | 0.173563 | 0.062805 | 0.157063 | 0.186405 |
| 4 | 0.081906 | 0.051769 | 0.078705 | 0.091911 |
| 5 | 0.148876 | 0.057298 | 0.138020 | 0.152134 |
| 6 | 0.169497 | 0.044303 | 0.159416 | 0.173816 |
| 7 | 0.138720 | 0.107130 | 0.133612 | 0.149857 |
| 8 | 0.412488 | 0.205264 | 0.385704 | 0.427975 |
| 9 | 0.284696 | 0.123177 | 0.261857 | 0.299352 |

Interpretation:

1. The uniform `10.00%` restored-input clone accuracy in Section 10 was caused
   by an input-normalization bug, not by the attack or model alone.
2. After correction, known-client clone accuracy no longer collapses to chance:
   baseline known-client reaches `30.00%`, and P&B known-client reaches
   `50.00%` on the ten selected samples.
3. The unknown-client attacks remain near chance, which is plausible because
   the attacker must learn both the recovered inputs and a clone client.
4. These corrected numbers are not directly identical to older paper numbers
   because this rerun uses the corrected stem split `[B, 32, 32, 32]` and the
   newly trained corrected checkpoints.

## 12. Final Attack-Objective Correction: Remove Image L2 Prior

**Date:** 2026-06-14  
**SLURM job:** `41214`  
**Remote run directory:** `/data03/home/yongzhiwang/develop/partition_and_blocking_20260604/attack_runs/l2zero_tv01_norm_converge_20260614_232603/`  
**Local output directory:** `partition_and_blocking_20260604/attack_runs/l2zero_tv01_norm_converge_20260614_232603/`

Section 11 fixed the input-normalization mismatch, but still used the previous
default attack regularization `lambda_l2 = 1.0`. That term penalizes the
magnitude of the reconstructed image itself:

```text
loss = smashed_feature_mse + lambda_tv * TV(x_hat) + lambda_l2 * mean(x_hat^2)
```

This is not the objective described in the paper's UnSplit summary, which uses
feature-space MSE plus a Total Variation regularizer. With `lambda_l2 = 1.0`,
the optimizer can reduce the total loss by pushing `x_hat` toward low pixel
magnitudes, producing dark reconstructions and artificially low
restored-input clone accuracy. A local known-client probe confirmed this:
setting `lambda_l2 = 0` while keeping normalized model inputs recovered the
baseline images with average pixel MSE below `0.001` and restored-input clone
accuracy of `90.00%`.

The corrected attack scripts now default to `lambda_l2 = 0.0` and print both
regularization weights in the Slurm logs. The rerun below keeps `lambda_tv =
0.1`, uses CIFAR-10 normalization before every model/client forward pass, and
uses the corrected checkpoints from Section 7.

Common attack settings:

| Setting | Value |
| ------- | ----: |
| `main-iters` | 100000 |
| `input-iters` | 100 |
| `model-iters` | 100 |
| `input-change-tol` | 0.0001 |
| `main-convergence-patience` | 5 |
| `min-main-iters` | 50 |
| `lambda-tv` | 0.1 |
| `lambda-l2` | 0 |
| CUDA required | Yes |
| Number of target images | 10 |

Final corrected attack results for the corrected-stem models:

| Attack | Client Knowledge | Device Reported | Smashed Tensor | Protected Region | Observed Ratio | Converged Iteration | Final Smashed MSE | Average Pixel MSE | Restored-Input Clone Accuracy |
| ------ | ---------------- | --------------- | -------------- | ---------------- | -------------: | ------------------: | ----------------: | ----------------: | ----------------------------: |
| Baseline MobileNetV1 full-smashed | Known | CUDA | `(10, 32, 32, 32)` | None | 1.0000 | 50 | 0.000401 | 0.001646 | 90.00% (9/10) |
| Baseline MobileNetV1 full-smashed | Unknown | CUDA | `(10, 32, 32, 32)` | None | 1.0000 | 50 | 0.004466 | 0.069509 | 10.00% (1/10) |
| Corrected P&B blocked-smashed | Known | CUDA | `(10, 32, 32, 32)` | `h=11:21, w=11:21` | 0.9023 | 50 | 0.000309 | 0.002729 | 90.00% (9/10) |
| Corrected P&B blocked-smashed | Unknown | CUDA | `(10, 32, 32, 32)` | `h=11:21, w=11:21` | 0.9023 | 79 | 0.005753 | 0.071553 | 10.00% (1/10) |

Per-image pixel MSE:

| Class Index | Baseline Known | Baseline Unknown | P&B Known | P&B Unknown |
| ----------: | -------------: | ---------------: | --------: | ----------: |
| 0 | 0.001400 | 0.061832 | 0.002028 | 0.057913 |
| 1 | 0.002751 | 0.108331 | 0.006461 | 0.112924 |
| 2 | 0.000971 | 0.045629 | 0.001861 | 0.046843 |
| 3 | 0.001255 | 0.038679 | 0.001565 | 0.041557 |
| 4 | 0.000132 | 0.043672 | 0.000378 | 0.053494 |
| 5 | 0.001235 | 0.061797 | 0.001259 | 0.065400 |
| 6 | 0.001051 | 0.034334 | 0.000769 | 0.036037 |
| 7 | 0.004160 | 0.127030 | 0.006791 | 0.134267 |
| 8 | 0.001534 | 0.108644 | 0.003104 | 0.104137 |
| 9 | 0.001972 | 0.065145 | 0.003075 | 0.062958 |

Interpretation:

1. The all-`10.00%` restored-input clone accuracy in Section 10 had two causes:
   an input-normalization mismatch and an overly strong image L2 prior.
2. With both fixes applied, the known-client attacks no longer collapse. The
   baseline known-client attack reaches `90.00%` RICA, and the corrected P&B
   known-client attack also reaches `90.00%` RICA on the ten selected samples.
3. For the corrected-stem architecture, P&B still increases reconstruction MSE
   relative to the baseline in both threat settings, but the effect is much
   smaller than the older paper-era numbers: `0.001646 -> 0.002729` for
   known-client and `0.069509 -> 0.071553` for unknown-client.
4. These final corrected-stem results should not be mixed with the older
   `main.pdf` tables without also updating the model-description text. The
   older paper experiment used a different P&B split/architecture and reported
   larger absolute MSE gaps. The corrected-stem model exposes a shallower and
   more redundant `(32, 32, 32)` stem representation, so missing only the
   central `10 x 10` spatial block does not reproduce the older paper's
   `0.018 -> 0.061` known-client MSE pattern.
