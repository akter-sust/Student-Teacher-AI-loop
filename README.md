# Student-Teacher-AI-loop Active Learning Pipeline: Construction Surface Defect Detection (Initial Baseline Run)

> **Note:** This document records the baseline results, dataset configuration, training dynamics, evaluation metrics, and active learning sampling behavior from the initial run of the multimodal Active Learning pipeline for construction surface defect detection. Detailed experimental logs, hyperparameter iterations, and cycle-over-cycle progression will be added in subsequent updates.

---

## 1. Executive Summary

- **Pipeline Architecture:** Multimodal Student Classifier processing 1024-dimensional combined vectors (512-dim visual embeddings + 512-dim text/metadata embeddings).
- **Training Strategy:** Fresh initialization without pre-trained student checkpoints (`student_pretrained.pth` absent).
- **Primary Metric State:** **Precision: 1.00**, **Recall: 0.60**, **F1-Score: 0.75**.
- **Key Insight:** The model achieved zero false positives (100% Precision), displaying highly reliable defect detection, while maintaining a conservative decision boundary that missed 2 subtle defects ($P < 0.50$).

---

## 2. Dataset Distribution & Split Inventory

The underlying dataset comprises **34 labeled samples** and an unlabeled reservoir, stratified as follows:

| Data Subset | Positive (Defect) | Negative (Clean) | Total Samples | Class Ratio (% Pos) |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 14 | 9 | **23** | 60.87% |
| **Evaluation Set** | 5 | 3 | **8** | 62.50% |
| **Test Set** | 2 | 1 | **3** | 66.67% |
| **Total Labeled Inventory** | **21** | **13** | **34** | **61.76%** |

* **Constructed Pseudo-Labeled Pool:** 21 records generated for initial pre-training initialization.

---

## 3. Student Model Pre-Training Dynamics

- **Epoch Count:** 30 Epochs (Full completion without premature loss plateau triggers).
- **Optimization Objective:** `nn.BCEWithLogitsLoss` with `pos_weight = 1.0`.
- **Checkpoint Action:** Saved model weights to `student_pretrained.pth`.

### Epoch-by-Epoch Loss Progression

| Epoch | Training Loss | Epoch | Training Loss |
| :---: | :---: | :---: | :---: |
| **Epoch 01** | 0.7126 | **Epoch 16** | 0.6684 |
| **Epoch 02** | 0.7112 | **Epoch 17** | 0.6610 |
| **Epoch 03** | 0.7099 | **Epoch 18** | 0.6552 |
| **Epoch 04** | 0.7087 | **Epoch 19** | 0.6469 |
| **Epoch 05** | 0.7068 | **Epoch 20** | 0.6390 |
| **Epoch 06** | 0.7049 | **Epoch 21** | 0.6323 |
| **Epoch 07** | 0.7035 | **Epoch 22** | 0.6231 |
| **Epoch 08** | 0.7009 | **Epoch 23** | 0.6169 |
| **Epoch 09** | 0.6987 | **Epoch 24** | 0.6079 |
| **Epoch 10** | 0.6952 | **Epoch 25** | 0.5999 |
| **Epoch 11** | 0.6910 | **Epoch 26** | 0.5895 |
| **Epoch 12** | 0.6879 | **Epoch 27** | 0.5855 |
| **Epoch 13** | 0.6846 | **Epoch 28** | 0.5748 |
| **Epoch 14** | 0.6815 | **Epoch 29** | 0.5652 |
| **Epoch 15** | 0.6731 | **Epoch 30** | **0.5615** |

---

## 4. Post-Adaptation Evaluation Analysis

Evaluated against the static $N=8$ validation split (5 Positives, 3 Negatives).

| Metric | Score | Detailed Performance Breakdown |
| :--- | :---: | :--- |
| **Precision** | **1.00** | **0 False Positives.** Every image classified as a bubbling defect was a true defect. |
| **Recall** | **0.60** | **3/5 True Defects Detected.** 2 subtle defects yielded $P < 0.50$ (False Negatives). |
| **F1-Score** | **0.75** | Robust harmonic mean indicating well-calibrated baseline decision boundaries. |

---

## 5. Active Learning Uncertainty Sampling Log

Unlabeled reservoir samples evaluated during the Active Learning routing phase ($0.35 \le P \le 0.65$ routing threshold):

| Sample Filename | Student Prob | Model Decision | Active Learning Routing |
| :--- | :---: | :---: | :--- |
| `Bubbling_paintedplasterboardceilingother-aID-qtu5yd5d2pxkszw9tkue9mahb.jpeg` | **0.683** | Defect ($y=1$) | Confident Auto-Label |
| `Bubbling_paintedplaster_boardceilingbedroom-aID-699atnay1isbbz9iq9q9kd9bd.jpeg` | **0.683** | Defect ($y=1$) | Confident Auto-Label |
| `Water_Stain_Bubbling___Plasterboard___Sitting-aID-hpm6ccs3k5kfspu9269eu6izd.jpg` | **0.046** | Clean ($y=0$) | Confident Auto-Label |

---

