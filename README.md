# Student-Teacher Active Learning VLM Pipeline for Surface Defect Detection

An end-to-end active learning and knowledge distillation framework leveraging Vision-Language Models (VLMs) to detect construction surface defects—specifically focusing on **surface bubbling** in paint, plasterboard, and coatings.

The architecture combines a powerful **Teacher VLM** (Gemini 3.5 Flash via Google GenAI) that provides zero-shot/few-shot pseudo-labeling with structured Pydantic validation, and a lightweight **Student VLM** (Qwen2-VL-2B-Instruct) fine-tuned using Parameter-Efficient Fine-Tuning (PEFT / LoRA).

---

## 📌 Features & Architecture

* **Teacher-Student Distillation:** Generates high-confidence pseudo-labels using Gemini 3.5 Flash to supervise student model training.
* **LoRA Fine-Tuning:** Adapts `Qwen2-VL-2B-Instruct` efficiently using `peft` with target modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Structured Output Parsing:** Enforces structured JSON output and validates model responses using **Pydantic** (`DefectPrediction` and `StudentDefectResponse`).
* **Data Leakage Prevention:** Groups original images and their augmented versions (e.g., rotated variants) into single splits so no related images cross between Train, Evaluation, and Test splits.
* **Active Learning & Uncertainty Routing:** Routes low-confidence predictions ($0.35 \le P \le 0.65$) back to the teacher model or human-in-the-loop reviewers.
* **Full Pipeline Lifecycle:** Automatically executes baseline evaluation, teacher-supervised pseudo-labeling, LoRA fine-tuning, checkpoint saving, model reloading, and post-adaptation evaluation.

---

## 📂 Repository Structure

```text
synctech/
├── train-bubbling/           # Positive defect image samples (and rotated augmentations). This is not available in the repo.
├── hard-negatives-bubbling/  # Negative/hard-negative image samples. This is not available in the repo.
├── app_vlm.py                # VLM based student model
├── app_dl.py                 # Deep Learning based student model
├── README.md                 # Main documentation
├── README_dl.md              # Deep learning benchmark documentation
├── requirements_vlm.txt      # Python dependencies for VLM pipeline
├── requirements_dl.txt       # Dependencies for Deep learning baseline
├── resize_images.py          # Utility script for standardizing image resolutions
├── synthetic_data.py         # Utility script for generating rotated data augmentations
├── results_vlm.txt           # Benchmark output logs and pipeline results for VLM based student model
└── TEST_REQUIREMENTS.md      # Testing and validation specifications
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.12 installed with CUDA support.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements_vlm.txt
```

### 2. Environment Variables
Set your Gemini API key and optional cache path configuration before running the execution script:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

---

## 🚀 Running the VLM Pipeline

To execute the entire end-to-end pipeline (Data loading, Zero-shot evaluation, Teacher pseudo-labeling, LoRA training, and Post-adaptation evaluation):

```bash
python app_vlm.py
```

### Pipeline Execution Steps:
1. **Dataset Splitting:** Categorizes images into train, eval, and test sets using prefix mapping (`eval_`, `test_`) while retaining rotated augmentations within the same split to avoid data leakage.
2. **Zero-Shot Evaluation:** Measures pre-training baseline precision, recall, and F1-score on `Qwen2-VL-2B-Instruct`.
3. **Teacher Supervision:** Queries `gemini-3.5-flash` for unlabelled samples and filters pseudo-labels with confidence $\ge 0.75$.
4. **Student LoRA Training:** Fine-tunes target projection layers for 3 epochs using Hugging Face `Trainer`.
5. **Checkpoint Reloading:** Saves adapter weights to `./final_bubbling_vlm_lora`, frees CUDA cache, and reloads saved weights for unbiased evaluation.
6. **Active Learning Inference:** Runs inference on unseen test samples and flags uncertain predictions ($0.35 \le P \le 0.65$).

---

## 📊 Experimental Results

Evaluating performance on the surface bubbling dataset:

| Model Stage | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Zero-Shot Baseline (Qwen2-VL-2B)** | 0.48 | 1.00 | 0.65 |
| **Post-Adaptation (LoRA Fine-Tuned)** | **1.00** | **1.00** | **1.00** |

* **Analysis: Reasons for Perfect Model Performance (1.00 F1-Score)**

   * **Text Context Leakage:** The prompt passes raw filename metadata (e.g., material types, identifiers, or prefix patterns) into the context string. Even with "bubbling" present in both classes, the model can exploit subtle text formatting cues instead of inspecting the visual pixels.
   * **Tiny Unique Sample Space:** The entire dataset relies on just 34 unique physical images (21 positive, 13 negative). With only a handful of distinct visual scenes in the test split, the 2B-parameter model easily memorizes every background texture and lighting condition.
   * **High Capacity vs. Simple Task:** Fine-tuning projection layers (`q_proj`, `v_proj`, etc.) via LoRA gives a powerful 2-billion parameter vision-language model massive capacity relative to a small binary task, enabling it to fit a perfect decision boundary for this narrow dataset.
   * **Teacher Label Consistency:** Ground-truth labels across all splits were pseudo-labeled by Gemini 3.5 Flash. The student model only has to compress and replicate the teacher's clean, deterministic rulebook rather than dealing with noisy human labels.

### Test Inference & Uncertainty Sampling Highlights
* The fine-tuned student model achieved **1.000 probability** on positive test samples (`test_Bubbling_ACM_Laundry...`) and correctly identified negative samples (`test_3-Bubbling_Staining...`).
* Rotation invariants held across all synthetic augmented test angles ($0^\circ, 15^\circ, 70^\circ, 105^\circ, 225^\circ, 345^\circ$, etc.).