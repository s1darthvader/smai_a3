# BanglaLekha Character Recognition

[cite_start]This repository contains the implementation of a custom **Convolutional Neural Network (CNN)** designed to recognize handwritten Bengali characters[cite: 13]. [cite_start]Developed for **SMAI Assignment 3**, the project focuses on training a compact, efficient model from scratch using the **BanglaLekha-Isolated** dataset[cite: 1, 16].

---

## 🚀 Live Demo
Experience the real-time recognition system here: 
[cite_start]**[Hugging Face Space](https://huggingface.co/spaces/siddharth2504/bangalekha)** [cite: 8]

---

## 📌 Project Overview
* [cite_start]**Model:** Custom 5-layer CNN (<500k parameters)[cite: 16, 60].
* [cite_start]**Dataset:** BanglaLekha-Isolated (84 classes, ~166,105 samples)[cite: 22, 23].
* [cite_start]**Performance:** **94.84%** validation accuracy on the full 84-class benchmark[cite: 96, 134].
* [cite_start]**Deployment:** Interactive Streamlit application for drawing and real-time Unicode prediction[cite: 14].

---

## 🏗️ Architecture & Training
[cite_start]The model utilizes a custom architecture where depth and regularization were optimized through systematic ablation[cite: 17, 50].

### Model Structure
[cite_start]Each convolutional block follows the pattern[cite: 50, 51, 52, 53, 54]:
`Conv(3x3) → BatchNorm → ReLU → MaxPool(2x2)`

[cite_start]Feature maps scale from 32 to 512 channels, followed by **Global Average Pooling (GAP)** and two fully connected layers with **Dropout (p=0.4)**[cite: 55, 56, 57, 58, 59].

### Training Protocol
* [cite_start]**Loss Function:** Cross-entropy[cite: 62].
* [cite_start]**Optimizer:** AdamW ($lr=0.001$, weight decay $= 10^{-4}$)[cite: 93].
* [cite_start]**Hardware:** PyTorch DDP on an HPC cluster ($512$ batch size per $GPU \times 4$ GPUs)[cite: 64].
* [cite_start]**Scheduling:** Linear warm-up followed by cosine annealing over 40 epochs[cite: 63, 65].

---

## 📊 Ablation Study Results
[cite_start]We investigated four design dimensions to identify the optimal configuration[cite: 79, 81]:

| Phase | Variable | Best Value | Best Val. Acc. (%) |
| :--- | :--- | :--- | :--- |
| **1** | Model Depth | **5 Layers** | 94.90% |
| **2** | Augmentation | **Enabled** | 94.84% |
| **3** | Optimizer | **AdamW** | 94.84% |
| **4** | Dropout | **0.4** | 94.84% |

---

## 🏷️ Labeling Methodology
[cite_start]The original dataset lacked Unicode ground-truth labels[cite: 68]. [cite_start]We implemented a **two-pass Gemini-assisted verification protocol** to map folder numbers to Bengali graphemes[cite: 69, 71]:
1.  [cite_start]**Pass 1:** Automated Unicode prediction via Gemini for one representative image per class[cite: 70].
2.  [cite_start]**Pass 2:** Independent repetition with fresh, distinct images to detect hallucinations[cite: 70].
3.  [cite_start]**Tie-break:** Majority voting for classes where initial passes disagreed (primarily complex conjunct characters)[cite: 70, 77].

---

## 📂 Repository Structure
* [cite_start]`main.py`: The **main train script** which includes the custom CNN architecture and the hardcoded Unicode mapping[cite: 75].
* `run.sh`: The **batch run script** used to execute training jobs.
* `app.py`: The Streamlit application code for the interactive demo.
* `FINAL_PRODUCTION_MODEL.pth`: The trained optimal model weights.
* `report.pdf`: The detailed technical report of the project.
* [cite_start]`confusion_matrix_best.png`: Visualization of model performance showing the diagonal structure of high accuracy[cite: 97, 99, 100].

---

## 📝 References
1. Biswas, M., et al. (2017). [cite_start]"Bangla Lekha-Isolated: A multi-purpose comprehensive dataset..."[cite: 138, 139].
2. Chowdhury, R. H., et al. (2026). [cite_start]"BornoViT: A novel efficient Vision Transformer..."[cite: 151, 152].

[cite_start]**Team:** The Last Jedi [cite: 7]
[cite_start]**Author:** Siddharth Singh [cite: 6]
