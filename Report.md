# Technical Deliverables & System Architecture

# Student-Teacher Active Learning VLM Pipeline for Surface Defect Detection

## Deliverables

### A. Short Design Note

* **Problem Approach:**
  * **Knowledge Distillation & Active Learning Loop:** Built a closed-loop VLM pipeline to detect visual surface defects (e.g., surface bubbling in paint/coatings).
  * **Teacher-Student Paradigm:** Leveraged a high-capacity teacher model (`gemini-3.5-flash`) for zero-shot pseudo-labeling and structured verification, distilling domain knowledge into a compact student model (`Qwen2-VL-2B-Instruct`).
  * **Native Dynamic Resolution:** Unlike fixed-resolution vision models, Qwen2-VL handles arbitrary aspect ratios without cropping or visual distortion, crucial for fine-grained localized surface defect detection.
  * **Multimodal Rotary Position Embedding (M-RoPE):** Features 3D position embeddings to seamlessly bind visual spatial context with text tokens.
  * **Extreme Efficiency & Edge Readiness:** At ~2B parameters, it allows low-latency inference and low-VRAM LoRA fine-tuning (~8-12 GB), making it deployable on edge inspection devices.
  * **Strong Perception Metrics:** Achieves top-tier benchmarks in document and fine-grained visual perception (90.1% DocVQA, 79.7% TextVQA).
  * **Parameter-Efficient Adaptation:** Used Low-Rank Adaptation (LoRA / PEFT) to fine-tune key vision and language projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) of the student VLM while keeping overall compute footprint low.

* **Assumptions Made:**
  * **Structured JSON Outputs:** Expected the VLM to produce strictly formatted JSON responses containing binary classification flags (`bubbling`: `YES/NO`) and confidence scores ($0.0 \text{ to } 1.0$).
  * **Filename Metadata Exploitation:** Assumed metadata extracted directly from filenames (e.g., surface material, location context, and damage identifiers) provides rich contextual signals to improve defect reasoning.
  * **Data Invariance:** Assumed rotational invariance applies to surface defect photos, allowing synthetic rotation augmentations to simulate real-world field capture angles.

* **Handling Limited or Noisy Data:**
  * **Synthetic Data Generation:** Expanded a small real-world set (21 positive, 13 negative samples) into 740 augmented samples via multi-angle rotational transformations ($0^\circ$ to $360^\circ$).
  * **Data Leakage Prevention:** Ensured original images and all their rotated variants were strictly grouped into the *same* split (Train, Eval, or Test) so no synthetic pair crossed evaluation boundaries.
  * **Confidence Filtering:** Applied a confidence threshold ($\ge 0.75$) on teacher pseudo-labels, discarding ambiguous labels before passing them to the student training set.
  * **Pydantic Validation & Fallbacks:** Enforced strict schema validation using `Pydantic` to normalize messy or non-standard model outputs, backed by a deterministic regex fallback parser.

* **Teacher Model Utilization:**
  * **Zero-Shot / Few-Shot Labeling:** Used Gemini 3.5 Flash as an automated supervisor to label raw, unannotated dataset images.
  * **Uncertainty Routing (Active Learning):** Integrated an uncertainty band ($0.35 \le P \le 0.65$) during inference. Unconfident student model predictions are routed back to the teacher or flagged for human review.

* **Memory Management & Batching Optimization:**
  * **Dynamic Image Resolution Constraint:** Configured `min_pixels` and `max_pixels` image bounds to control visual token sequence lengths, preventing Out-Of-Memory (OOM) errors on large image inputs.
  * **Custom Data Collator (`Qwen2VLDataCollator`):** Implemented a custom collator that handles vision token masking by explicitly target-masking special tokens (`<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>`) and standard padding tokens (`labels == pad_token_id`) to `-100`, ensuring backpropagation calculates loss strictly on text generation outputs.
  * **Training Hyperparameter Tuning:** Experimented across `per_device_train_batch_size` configurations while optimizing throughput using `num_train_epochs=3`, `bf16=True`, `dataloader_num_workers=4`, and `dataloader_pin_memory=True`.

* **Progress Measurement:**
  * **Metric Suite:** Evaluated Precision, Recall, and F1-score across distinct splits.
  * **Comparative Baseline:** Benchmark zero-shot performance of the base student model against post-adaptation performance after LoRA fine-tuning.
  * **Inference Tracking:** Analyzed student predicted probabilities on unseen test images and out-of-distribution (OOD) rotated angles.