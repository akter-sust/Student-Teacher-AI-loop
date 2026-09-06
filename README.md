# Student-Teacher-AI-loop Active Learning Pipeline: Construction Surface Defect Detection (Initial Baseline Run)

> **Note:** This document records the baseline results, dataset configuration, training dynamics, evaluation metrics, and active learning sampling behavior from the initial run of the multimodal Active Learning pipeline for construction surface defect detection. Detailed experimental logs, hyperparameter iterations, and cycle-over-cycle progression will be added in subsequent updates.

---

## 2. Executive Summary

- **Pipeline Architecture:** Multimodal Student Classifier processing 1024-dimensional combined vectors (512-dim visual embeddings + 512-dim text/metadata embeddings).
- **Training Strategy:** Fresh initialization without pre-trained student checkpoints (`student_pretrained.pth` absent).
- **Primary Metric State:** **Precision: 1.00**, **Recall: 0.60**, **F1-Score: 0.75**.
- **Key Insight:** The model achieved zero false positives (100% Precision), displaying highly reliable defect detection, while maintaining a conservative decision boundary that missed 2 subtle defects ($P < 0.50$).

---

## 3. Dataset Overview & Data Splitting
The dataset consists of multimodal samples categorized into positive and negative classes.

* **Total Samples:** 34 images
  * **Positive Images:** 21
  * **Negative Images:** 13
* **Data Partitioning:**
  * **Training Set:** 23 samples (14 Positive, 9 Negative)
  * **Evaluation Set:** 8 samples (5 Positive, 3 Negative)
  * **Test Set:** 3 samples (2 Positive, 1 Negative)

---

## 4. Zero-Shot Baseline Evaluation
Prior to distillation and fine-tuning, zero-shot evaluation was performed on the evaluation dataset.

* **Precision:** 0.00
* **Recall:** 0.00
* **F1-Score:** 0.00

---

## 5. Teacher Supervision & Pseudo-Labeling
* **Teacher Model:** Large Language Model / Multimodal Teacher (Gemini)
* **Pseudo-Label Generation:** Teacher labels extracted for all 23 training set samples.
* **Result:** Constructed a supervised pseudo-labeled dataset of 23 records for student model distillation.

---

## 6. Student Model Training Dynamics
The student model was trained from scratch over 30 epochs using loss feedback from teacher pseudo-labels.

### Training Loss Progress
| Epoch | Loss | Epoch | Loss | Epoch | Loss |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.6923 | **11** | 0.6216 | **21** | 0.4117 |
| **2** | 0.6881 | **12** | 0.6080 | **22** | 0.3836 |
| **3** | 0.6834 | **13** | 0.5899 | **23** | 0.3588 |
| **4** | 0.6782 | **14** | 0.5755 | **24** | 0.3407 |
| **5** | 0.6736 | **15** | 0.5568 | **25** | 0.2935 |
| **6** | 0.6682 | **16** | 0.5362 | **26** | 0.2706 |
| **7** | 0.6609 | **17** | 0.5158 | **27** | 0.2456 |
| **8** | 0.6524 | **18** | 0.4851 | **28** | 0.2164 |
| **9** | 0.6438 | **19** | 0.4698 | **29** | 0.1942 |
| **10** | 0.6343 | **20** | 0.4388 | **30** | **0.1574** |

* **Checkpoint Saved:** `student_pretrained.pth`

---

## 7. Post-Adaptation Evaluation Performance

| Metric | Zero-Shot Baseline | Post-Training Adaptation | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.00 | **1.00** | +1.00 |
| **Recall** | 0.00 | **0.60** | +0.60 |
| **F1-Score** | 0.00 | **0.75** | +0.75 |

---

## 8. Active Learning & Uncertainty Sampling
Uncertainty sampling demo on unannotated/candidate samples:

1. **Sample 1:** `Bubbling_paintedplasterboardceilingother-aID-qtu5yd5d2pxkszw9tkue9mahb.jpeg`
   * **Predicted Student Probability:** `0.898`
2. **Sample 2:** `Bubbling_paintedplaster_boardceilingbedroom-aID-699atnay1isbbz9iq9q9kd9bd.jpeg`
   * **Predicted Student Probability:** `0.831`
3. **Sample 3:** `Water_Stain_Bubbling___Plasterboard___Sitting-aID-hpm6ccs3k5kfspu9269eu6izd.jpg`
   * **Predicted Student Probability:** `0.197`
   