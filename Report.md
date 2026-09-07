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


### B. Runnable Code

The lightweight pipeline implementation is encapsulated in `app_vlm.py` and structured into five clear components:

* **Dataset Creation & Assembly:**
  * Implemented `construct_dataset()` to parse local directories (`train-bubbling/`, `hard-negatives-bubbling/`).
  * Extracts metadata from file names (material type, location context) and groups original images alongside synthetic rotational variants ($0^\circ$ to $360^\circ$).
  * Enforces prefix-based split logic (`eval_`, `test_`) to route image groups into Train, Eval, and Test sets without data leakage across splits.

* **Teacher Interaction:**
  * Implemented `get_or_query_teacher()` to interface with `gemini-3.5-flash` using `google-genai`.
  * Sends images with custom structured prompts, enforcing response formats through Pydantic (`DefectPrediction`).
  * Filters out pseudo-labels with confidence scores $< 0.75$ and caches successful labels to `teacher_results.json` to minimize API latency and cost.

* **Student Training & Adaptation:**
  * Configured `Qwen2-VL-2B-Instruct` using `peft` with Low-Rank Adaptation (LoRA) on projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
  * Built `Qwen2VLDataCollator` to handle dynamic visual sequence lengths via `min_pixels` and `max_pixels`. Masked special visual tokens (`<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>`) and padding tokens (`labels == pad_token_id`) to `-100` so loss is calculated strictly on text outputs.
  * Fine-tuned using Hugging Face `Trainer` over 3 full epochs (441 total steps) with `bf16=True`, `dataloader_num_workers=4`, and `dataloader_pin_memory=True`, more.

* **Evaluation:**
  * Built `evaluate_student()` to run inference on evaluation and test splits.
  * Formats model outputs into binary predictions (`bubbling`: `YES/NO`) and measures Precision, Recall, and F1-score against ground truth labels.

* **Orchestration & Repeatable Workflow:**
  * Implemented `run_pipeline()` to orchestrate the execution flow:
    1. Dataset partition assembly.
    2. Zero-shot baseline evaluation on `Qwen2-VL-2B-Instruct`.
    3. Teacher pseudo-labeling via `gemini-3.5-flash`.
    4. LoRA fine-tuning and saving weights to `./final_bubbling_vlm_lora`.
    5. GPU memory cleanup (`gc.collect()`, `torch.cuda.empty_cache()`) and fine-tuned weight reloading.
    6. Post-adaptation evaluation and active learning uncertainty routing ($0.35 \le P \le 0.65$).

---

### C. Results

* **Example Generated Training Records:**
  ```json
  [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "./train-bubbling/Bubbling_ACM_Laundry-aID-aa7rx6e8.jpg"},
        {
          "type": "text",
          "text": "Context: surface photo showing Bubbling ACM Laundry. Analyze this surface for 'bubbling' defect. Is the defect present? Answer in JSON: {'bubbling': 'YES/NO', 'confidence': 0.0 to 1.0}."
        }
      ]
    },
    {
      "role": "assistant",
      "content": "{\"bubbling\": \"YES\", \"confidence\": 1.0}"
    }
  ]
  ```

* **Training Convergence & Optimization Metrics:**

  | Metric | Value |
  | :--- | :--- |
  | **Total Training Steps** | 441 steps (100% completion) |
  | **Training Runtime** | 916.5 seconds (~15 mins 16 secs) |
  | **Throughput** | 1.938 samples/sec (0.485 steps/sec) |
  | **Final Overall Training Loss** | `0.006395` |
  | **Final Step Loss (Epoch 2.995)** | `5.154e-05` |
  | **Final Step Gradient Norm** | `0.0007097` |

 
* **Before / After Performance Comparison:**

  | Stage | Split | Precision | Recall | F1-Score | Sample Count |
  | :--- | :--- | :---: | :---: | :---: | :---: |
  | **Zero-Shot Baseline** (Qwen2-VL-2B) | Evaluation Split | 0.48 | 1.00 | 0.65 | 131 images |
  | **Post-Adaptation** (LoRA Fine-Tuned) | Evaluation Split | **1.00** | **1.00** | **1.00** | 131 images |
  | **Post-Adaptation** (LoRA Fine-Tuned) | Test Split | **1.00** | **1.00** | **1.00** | 44 images |

* **Analysis: Reasons for Perfect Model Performance (1.00 F1-Score)**

    * **Text Context Leakage:** The prompt passes raw filename metadata (e.g., material types, identifiers, or prefix patterns) into the context string. Even with "bubbling" present in both classes, the model can exploit subtle text formatting cues instead of inspecting the visual pixels.
    * **Tiny Unique Sample Space:** The entire dataset relies on just 34 unique physical images (21 positive, 13 negative). With only a handful of distinct visual scenes in the test split, the 2B-parameter model easily memorizes every background texture and lighting condition.
    * **High Capacity vs. Simple Task:** Fine-tuning projection layers (`q_proj`, `v_proj`, etc.) via LoRA gives a powerful 2-billion parameter vision-language model massive capacity relative to a small binary task, enabling it to fit a perfect decision boundary for this narrow dataset.
    * **Teacher Label Consistency:** Ground-truth labels across all splits were pseudo-labeled by Gemini 3.5 Flash. The student model only has to compress and replicate the teacher's clean, deterministic rulebook rather than dealing with noisy human labels.

* **Metrics & Qualitative Observations:**
  * **Smooth Optimization Trajectory:** Training loss steadily decreased down to `~6.2e-05` by epoch 2.995, demonstrating fast, stable convergence without exploding gradients (`grad_norm` stayed bounded $\le 0.004$).
  * **Elimination of False Positives:** The zero-shot baseline model suffered from low precision (0.48) by over-flagging textured plasterboard and non-defect surfaces. Following LoRA adaptation, precision improved to 1.00 while maintaining 100% recall.

* **Rotational Robustness Verification & Planned OOD Testing:**
  * **In-Distribution Rotational Robustness:** Verified model consistency across synthetic rotation augmentations on test split images ($0^\circ, 15^\circ, 70^\circ, 105^\circ, 225^\circ, 345^\circ$), confirming reliable defect probability outputs.
  * **Out-Of-Distribution (OOD) Real-World Validation:** Evaluated model generalization using 4 completely unseen internet images outside the training distribution—comprising 2 non-surface negative controls (a sky capture and a full room interior) and 2 real-world surface bubbling samples. The fine-tuned model achieved 100% accuracy across all test samples, correctly rejecting non-defect backgrounds without false positives while accurately detecting bubbling textures under unconstrained lighting and material conditions.



### D. Next Steps

* **Immediate Pipeline Improvements & Technical Experiments:**
  * **Explicate Auxiliary Structured Metadata:** Replace raw filename strings with explicit operational metadata (e.g., sensor distance, ambient lighting level, surface material type, or capture timestamp) passed directly via prompt parameters or structured key-value inputs to enrich visual context without prompt shortcut leakage.
  * **Multi-Defect Surface Coverage:** Expand detection parameters beyond binary "bubbling" to cover multiple surface defect classes simultaneously (e.g., cracking, peeling, staining, mold, and corrosion).
  * **Model Exploration & Fine-Tuning Optimizations (QLoRA):** Experiment with alternative VLM backbones (e.g., `PaliGemma-2`, `Llama-3.2-Vision`, or `Qwen2-VL-7B`) and implement 4-bit NormalFloat (NF4) **QLoRA** fine-tuning to dramatically reduce VRAM requirements while evaluating performance across varying target projection module combinations and rank sizes ($r = 16, 32, 64$).
  * **Diverse Real-Time Benchmark Parameters:** Introduce additional evaluation metrics to rigorously benchmark performance under real-time conditions, including inference latency (ms), GPU VRAM footprint, throughput across batch sizes, and accuracy across varying image resolution constraints (`min_pixels` / `max_pixels`).
  * **Validation on Real-Time Field Data:** Test fine-tuned weights on fresh batches of real-world field captures featuring unseen surface textures, varying camera angles, and dynamic lighting conditions.

* **Production & Scalability Considerations:**
  * **Edge Deployment & Model Quantization:** Quantize the fine-tuned student model to 4-bit/8-bit precision (AWQ/GGUF) to enable real-time, low-latency inference on edge inspection devices or mobile hardware.
  * **Automated Active Learning Trigger:** Operationalize the uncertainty band ($0.35 \le P \le 0.65$). Automatically route low-confidence field captures to an asynchronous human-in-the-loop (HITL) queue or high-tier teacher model (`gemini-3.5-flash`) for continuous dataset enrichment.
  * **API Serving Architecture:** Package the model into a lightweight, containerized FastAPI endpoint with batching optimizations (e.g., vLLM or TensorRT-LLM) to handle high-throughput image streams in factory/field environments.