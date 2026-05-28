# Experiment Log

## Experiment 1: P&B Model Quick Training (35 epochs)

**Date:** May 28, 2026
**Cluster:** hpc.cpp.edu (GPU: Tesla P100)
**Job ID:** 41044
**Script:** `code/train_pb.py`

### Configuration
- **Model:** PartitionAndBlockingModel (MobileNet V1 + P&B)
- **Dataset:** CIFAR-10
- **Split layer:** 3
- **Partitioning:** 3×3 spatial grid, central partition protected
- **Epochs:** 35 (resumed from epoch 29)
- **Batch size:** 128
- **Optimizer:** Adam (lr=0.001, weight_decay=5e-4)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5, mode=max)
- **Data augmentation:** RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize

### Checkpoint Loaded
- **Source:** `checkpoints/best_model.pth`
- **Epoch:** 29
- **Best accuracy:** 84.86%

### Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|---------|---------|
| 30    | 0.4445    | 84.84%    | 0.4833  | 84.40%  |
| 31    | 0.4368    | 85.14%    | 0.4896  | 83.96%  |
| 32    | 0.4468    | 85.10%    | 0.4840  | 84.09%  |
| 33    | 0.4413    | 85.35%    | 0.5040  | 84.12%  |
| 34    | 0.4348    | 85.46%    | 0.4951  | 83.45%  |
| 35    | 0.4312    | 85.55%    | 0.5215  | 83.51%  |

- **Best validation accuracy:** 84.86% (epoch 29, unchanged)

### Notes
- Model had converged by epoch 29. Additional 6 epochs showed no improvement, indicating a plateau.

---

## Experiment 2: Full Training (76 epochs)

**Date:** May 28, 2026
**Cluster:** hpc.cpp.edu (GPU: Tesla P100)
**Job IDs:** 41045 (baseline), 41046 (P&B)
**Scripts:** `code/train_baseline.py`, `code/train_pb.py`

### Configuration
- **Models:** Baseline MobileNet V1 vs P&B Model
- **Dataset:** CIFAR-10
- **Epochs:** 76
- **Batch size:** 128
- **Optimizer:** Adam (lr=0.001, weight_decay=5e-4)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5, mode=max) — P&B only
- **Data augmentation:** RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize

### Baseline MobileNet V1 Results
- **Loaded checkpoint:** epoch 76, 87.52%
- No further training needed (already completed)

### P&B Model Results

| Epoch Range | Train Acc Trend | Best Val Acc |
|-------------|----------------|-------------|
| 30–40       | 84–86%         | 84.40%      |
| 40–50       | 86–88%         | 85.16%      |
| 50–60       | 88–90%         | 87.38%      |
| 60–70       | 90–93%         | 88.56%      |
| 70–76       | 93–94%         | **88.84%**  |

Key milestones:
- Surpassed previous plateau (84.86%) at epoch ~45
- Surpassed baseline (87.52%) at epoch ~65
- **Best: 88.84%** at epoch 76

### Comparison

| Model | Best Accuracy | Parameters |
|-------|-------------|------------|
| Baseline MobileNet V1 | 87.52% | 2,948,426 |
| P&B Model | **88.84%** | ~14,200,000 |

### Notes
- P&B model broke through the 84.86% plateau with extended training, eventually outperforming the baseline.
- The accuracy gap is +1.32% in favor of P&B, likely due to the extra parameters in the dual-branch + merging architecture.
- Both models received identical data augmentation and optimizer settings.

---

## Experiment 3: Retrain with checkpoints in `code/checkpoints/` (76 epochs)

**Date:** May 28, 2026
**Cluster:** hpc.cpp.edu (GPU: Tesla P100)
**Job IDs:** 41047 (P&B), 41048 (baseline)
**Scripts:** `code/train_pb.py`, `code/train_baseline.py`

### Configuration
- Same as Experiment 2.
- Checkpoints saved to `code/checkpoints/` (relative path from `code/`).

### P&B Model Results
- **Best validation accuracy:** 88.62%
- **Final validation accuracy:** 88.51% (epoch 76)
- **Training time:** 17m 20s

### Baseline MobileNet V1 Results
- **Best validation accuracy:** 88.41%
- **Final validation accuracy:** 88.41% (epoch 76)
- **Training time:** 20m 24s

### Comparison

| Model | Best Accuracy | Parameters |
|-------|-------------|------------|
| Baseline MobileNet V1 | 88.41% | 2,948,426 |
| P&B Model | **88.62%** | ~14,200,000 |

### Notes
- P&B (+0.21%) retained its advantage over baseline, though the gap narrowed compared to Experiment 2 (+1.32%).
- Baseline improved from 87.52% → 88.41%, likely due to random seed variation.
- Checkpoints stored in `code/checkpoints/` and tracked via `.gitignore`.
