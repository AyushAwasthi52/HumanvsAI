# Level 4 Model Training Report: AI vs Human Text Detection

**Status**: 🚀 **Training Active (In Progress)**  
**Target Output Directory**: [`./models_level4`](file:///home/ayush/Code/AI/text_detector/models_level4)  
**Task Log File**: [Live Task Log](file:///home/ayush/.gemini/antigravity-ide/brain/d24c301b-fc03-4c45-925e-34061acec8c2/.system_generated/tasks/task-165.log)

---

## 1. Executive Summary & Objective

Level 4 is the **High-Accuracy Project Submission Configuration**. It scales up the training volume to **15,000 training samples** and **2,500 validation samples** across multiple epochs using Bayesian Hyperparameter Optimization (Optuna) and Parameter-Efficient Fine-Tuning (LoRA).

The goal of this phase is to produce a robust, highly calibrated classifier with expected accuracy exceeding **97%** and ROC-AUC approaching **0.99**, while maintaining low false-positive rates on human text.

---

## 2. Experimental Setup & Hardware

| Parameter | Value |
| :--- | :--- |
| **GPU Accelerator** | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| **Compute Capability** | 8.6 (Ampere Architecture) |
| **Precision** | FP16 Mixed Precision |
| **Base Language Model** | `roberta-base` (12-layer Transformer, 768 hidden, 12 attention heads) |
| **PEFT Method** | LoRA (Low-Rank Adaptation) on `["query", "value", "key", "output.dense"]` |
| **Trainable Parameter Ratio** | ~1.2% (1,550,594 parameters trainable out of 126,197,764 total) |
| **Loss Function** | Custom **Focal Loss** ($\alpha=1.0, \gamma=2.0$) for hard-example focusing |

---

## 3. Dataset Configuration

- **Dataset Source**: `Hello-SimpleAI/HC3` (Cleaned & Stratified)
- **Active Training Split**: **15,000 samples** (Balanced: 7,500 Human / 7,500 AI)
- **Active Validation Split**: **2,500 samples** (Balanced: 1,250 Human / 1,250 AI)
- **Max Sequence Length**: 512 tokens (Truncation & Dynamic Padding)

---

## 4. Hyperparameter Search & Optimization Space (Optuna)

The run performs **3 Optuna trials** exploring the parameter space with aggressive Median Pruning:

```text
├── Learning Rate       : Uniform Log Float [1e-5, 3e-4]
├── Weight Decay        : Uniform Float [0.01, 0.15]
├── Warmup Steps        : Integer [5, 30]
├── LoRA Rank (r)       : Categorical [8, 16, 32]
├── LoRA Alpha (α)      : Categorical [16, 32, 64]
├── LoRA Dropout        : Uniform Float [0.05, 0.2]
└── Optimization Metric : ROC-AUC (Maximization)
```

### Training Hyperparameters:
- **Per-Device Batch Size**: 8
- **Gradient Accumulation Steps**: 2 (Effective Batch Size = 16)
- **Epochs per Trial**: 3
- **Learning Rate Scheduler**: Cosine Annealing with Warmup
- **Gradient Clipping**: `max_grad_norm = 1.0`
- **Early Stopping**: Patience = 2 epochs

---

## 5. How to Monitor Progress in Real Time

### A. View Live Terminal Logs
```bash
tail -f ~/.gemini/antigravity-ide/brain/d24c301b-fc03-4c45-925e-34061acec8c2/.system_generated/tasks/task-165.log
```

### B. Launch TensorBoard Visualization
```bash
source venv/bin/activate
tensorboard --logdir ./models_level4 --port 6006
```
Open `http://localhost:6006` in your browser to view live loss curves, learning rate decay, and validation metrics (accuracy, F1, precision, recall, and ROC-AUC).

---

## 6. Inference Command Post-Training

Once a trial checkpoint is produced inside `./models_level4/trial_<number>/checkpoint-<step>`, run predictions using:

```bash
python predict.py \
  --model ./models_level4/trial_0/checkpoint-<step> \
  --text "Your sample input paragraph goes here."
```
