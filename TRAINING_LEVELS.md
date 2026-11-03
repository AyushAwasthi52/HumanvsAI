# Training Levels & Run Arguments Guide

*Estimated times are based on an NVIDIA RTX 3050 Laptop GPU (6GB VRAM).*

---

### Level 1: Quick Sanity Check (~35 seconds)
> **Goal**: Rapid verification of code changes, pipeline health, and memory allocation.
- **Train Samples**: 600 | **Val Samples**: 150 | **Epochs**: 1 | **Trials**: 1
```bash
python 02_train.py \
  --n_trials 1 \
  --max_train_samples 600 \
  --max_val_samples 150 \
  --epochs 1 \
  --batch_size 8 \
  --output_dir ./models_level1
```

---

### Level 2: Fast Demo / Presentation Baseline (~3–5 minutes)
> **Goal**: Solid working baseline model (~93–95% accuracy) suitable for live demonstrations or quick experiments.
- **Train Samples**: 3,000 | **Val Samples**: 600 | **Epochs**: 2 | **Trials**: 1
```bash
python 02_train.py \
  --n_trials 1 \
  --max_train_samples 3000 \
  --max_val_samples 600 \
  --epochs 2 \
  --batch_size 8 \
  --output_dir ./models_level2
```

---

### Level 3: Balanced Hyperparameter Search (~15–20 minutes)
> **Goal**: Uses Optuna to compare 3 distinct hyperparameter combinations (learning rates, LoRA ranks, and warmup).
- **Train Samples**: 6,000 | **Val Samples**: 1,000 | **Epochs**: 2 | **Trials**: 3
```bash
python 02_train.py \
  --n_trials 3 \
  --max_train_samples 6000 \
  --max_val_samples 1000 \
  --epochs 2 \
  --batch_size 8 \
  --output_dir ./models_level3
```

---

### Level 4: High Accuracy Project Submission (~45–60 minutes)
> **Goal**: High-fidelity model (~97–98% accuracy) for formal project reports and evaluation tables.
- **Train Samples**: 15,000 | **Val Samples**: 2,500 | **Epochs**: 3 | **Trials**: 3
```bash
python 02_train.py \
  --n_trials 3 \
  --max_train_samples 15000 \
  --max_val_samples 2500 \
  --epochs 3 \
  --batch_size 8 \
  --output_dir ./models_level4
```

---

### Level 5: Full-Scale Production Benchmark (~3–5 hours)
> **Goal**: Complete fine-tuning on the entire dataset (all 63,034 training and 7,879 validation examples).
- **Train Samples**: 63,034 (All) | **Val Samples**: 7,879 (All) | **Epochs**: 3 | **Trials**: 5
```bash
python 02_train.py \
  --n_trials 5 \
  --epochs 3 \
  --batch_size 8 \
  --output_dir ./models_full
```

---

### Inference Command (Using Any Checkpoint)
```bash
python predict.py \
  --model <path_to_checkpoint_folder> \
  --text "Enter sample text to classify here"
```
