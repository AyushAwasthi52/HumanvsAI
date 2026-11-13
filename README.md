# Parameter-Efficient AI Text Detection

> **Detecting LLM-Generated Content via Deep LoRA, Focal Loss & Bayesian Optimization**  
> *Department of Computer Science and Engineering, Indian Institute of Information Technology Vadodara (IIIT Vadodara)*

---

## 👥 Project Team
* **Ayush Awasthi** (Roll No: `202451027` | `202451027@iiitvadodara.ac.in`)
* **Divyanshu Choudhary** (Roll No: `202451058` | `202451058@iiitvadodara.ac.in`)
* **Abhishek Pandey** (Roll No: `202451005` | `202451005@iiitvadodara.ac.in`)

---

## 🚀 Key Performance Highlights

| Metric | Full Fine-Tuning | Proposed Deep LoRA + Focal Loss | Efficiency Gain |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | 96.00% | **99.49%** | **+3.49%** |
| **ROC-AUC Score** | 0.9931 | **0.99998** | Near-perfect separation |
| **Recall on AI Text** | 97.00% | **99.96%** | 2,620 / 2,621 caught |
| **False Positive Rate** | 10.00% | **0.74%** | Minimal false alarms |
| **Trainable Parameters** | 126,197,764 | **1,550,594** | **-98.77% parameters** |
| **Peak Training VRAM** | > 11.2 GB | **4.2 GB** | **Fits on 6GB Laptop GPU** |
| **Adapter Disk Footprint** | 498.2 MB | **6.2 MB** | **-98.76% compression** |

---

## 📁 Repository Structure

```text
/home/ayush/Code/AI/
├── text_detector/                      # Core ML Engineering Repository
│   ├── 01_data_prep.py                 # Multi-stage HTML stripping, deduplication & Parquet split builder
│   ├── 02_train.py                     # Training engine: RoBERTa + Deep LoRA + Focal Loss + Optuna
│   ├── predict.py                      # Standalone CLI inference engine (auto-detects LoRA vs full checkpoints)
│   ├── benchmark_pretrained.py         # Zero-shot benchmarking against OpenAI RoBERTa baseline
│   ├── train_detector.py               # Classical full fine-tuning reference trainer
│   │
│   ├── data/                           # Snappy-compressed columnar Parquet dataset (78k records)
│   │   ├── train.parquet               # 63,034 samples (80%)
│   │   ├── val.parquet                 # 7,879 samples (10%)
│   │   └── test.parquet                # 7,880 samples (10%)
│   │
│   ├── models_full/                    # Full-scale Level 5 model checkpoints & TensorBoard telemetry
│   ├── models_level4/                  # High-accuracy Level 4 checkpoints (Optuna best trial)
│   ├── quick_train_models/             # Sanity smoke-test checkpoints
│   ├── training_logs/                  # Experiment logging records
│   │
│   ├── FULL_PROJECT_DOCUMENTATION.md   # Exhaustive technical documentation & architecture review
│   ├── LEVEL_4_TRAINING_REPORT.md      # Training telemetry & experimental log
│   ├── TRAINING_LEVELS.md              # 5-tier training configuration ladder (L1 to L5)
│   ├── requirements.txt                # Python package dependencies
│   ├── venv/                           # Dedicated Python virtual environment
│   └── README.md                       # Core repo documentation
│
├── paper/                              # IEEE Signal Processing Letters Package
│   ├── main.tex                        # 5-page IEEE SPL LaTeX source (Overleaf default)
│   ├── manuscript.tex                  # Identical IEEE LaTeX submission copy
│   ├── overleaf_manuscript.zip         # 1-click import zip for Overleaf (main.tex + fig1.png + fig2.png)
│   ├── fig1.png                        # Architectural diagram (Transformer backbone + LoRA adapters)
│   ├── fig2.png                        # End-to-end data preparation and Optuna pipeline
│   ├── architecture_flowchart_final.png # Master high-res architecture flowchart
│   └── pipeline_flowchart_final.png    # Master high-res pipeline flowchart
│
├── presentation/                       # Presentation & Defense Deliverables
│   ├── AI_Text_Detection_Presentation.pptx # Native 10-slide academic PowerPoint deck
│   ├── AI_Text_Detection_Presentation.pdf  # PDF preview & printout of presentation
│   ├── PRESENTATION_SCRIPT_HINGLISH.md     # 10-15 minute conversational presentation script with Q&A defense
│   └── build_human_deck.py             # Standalone Python script to regenerate PPTX
│
└── archive/                            # Refactored Archive (Drafts, scratch scripts & legacy notes)
    ├── Semester_Project_Final_Report.docx
    ├── AI_vs_Human_Benchmark_Report_Highly_Detailed.docx
    ├── Semester_Project_Final_Report_copy.docx
    ├── build_deck.py                   # Initial dark-mode presentation script
    ├── create_presentation.py          # Initial scratch script
    ├── test_human_style.py             # Table styling test script
    ├── architecture_flowchart.png      # Draft flowchart
    ├── pipeline_flowchart.png          # Draft flowchart
    ├── slide_img-*.png                 # Temporary rendered slide preview images
    └── info.txt                        # Legacy scratch note
```

---

## ⚡ Quickstart: Running Inference

To classify an arbitrary text string using your best trained LoRA checkpoint:

```bash
cd /home/ayush/Code/AI/text_detector
source venv/bin/activate

python predict.py \
  --model ./models_full/trial_0/checkpoint-3940 \
  --text "Artificial intelligence has profoundly transformed digital communication."
```

### Sample Output:
```text
Loading model and tokenizer: ./models_full/trial_0/checkpoint-3940
Detected LoRA/PEFT adapter checkpoint.

--- Prediction Results ---
Input Text: "Artificial intelligence has profoundly transformed digital communication..."
Prediction: AI Generated
Confidence (Human): 0.12%
Confidence (AI):    99.88%
```

---

## 📑 Deliverable Links
* **Paper Package**: [paper/](file:///home/ayush/Code/AI/paper/)
  * [main.tex](file:///home/ayush/Code/AI/paper/main.tex)
  * [overleaf_manuscript.zip](file:///home/ayush/Code/AI/paper/overleaf_manuscript.zip)
* **Presentation Package**: [presentation/](file:///home/ayush/Code/AI/presentation/)
  * [AI_Text_Detection_Presentation.pptx](file:///home/ayush/Code/AI/presentation/AI_Text_Detection_Presentation.pptx)
  * [AI_Text_Detection_Presentation.pdf](file:///home/ayush/Code/AI/presentation/AI_Text_Detection_Presentation.pdf)
  * [PRESENTATION_SCRIPT_HINGLISH.md](file:///home/ayush/Code/AI/presentation/PRESENTATION_SCRIPT_HINGLISH.md)
* **Technical Documentation**:
  * [FULL_PROJECT_DOCUMENTATION.md](file:///home/ayush/Code/AI/text_detector/FULL_PROJECT_DOCUMENTATION.md)
  * [LEVEL_4_TRAINING_REPORT.md](file:///home/ayush/Code/AI/text_detector/LEVEL_4_TRAINING_REPORT.md)
  * [TRAINING_LEVELS.md](file:///home/ayush/Code/AI/text_detector/TRAINING_LEVELS.md)
