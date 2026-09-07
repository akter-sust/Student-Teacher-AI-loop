import os
# Environment configurations
BASE_CACHE = "/mnt/data/cache"
os.environ["HF_HOME"] = f"{BASE_CACHE}/huggingface"
os.environ["HF_HUB_CACHE"] = f"{BASE_CACHE}/huggingface/hub"
os.environ["TMPDIR"] = f"{BASE_CACHE}/tmp"
os.environ["TRITON_CACHE_DIR"] = f"{BASE_CACHE}/triton"
os.environ["TORCH_HOME"] = f"{BASE_CACHE}/torch"

for path in ["huggingface/hub", "tmp", "triton", "torch"]:
    os.makedirs(f"{BASE_CACHE}/{path}", exist_ok=True)

os.environ["TORCH_NATIVE_TRITON_SEARCH_PATH"] = "0"
os.environ["TRITON_DISABLE"] = "1"

import glob
import json
from pathlib import Path
import re
import gc
from PIL import Image
from google import genai
from google.genai import types
from peft import LoraConfig, PeftModel, get_peft_model
from pydantic import BaseModel, Field, field_validator
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    Trainer,
    TrainingArguments,
)
import wordninja



# ---------------------------------------------------------------------------
# 1. Configuration & Global Setup
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONCEPT_NAME = "bubbling"
CONCEPT_DEF = (
    "Bubbling appears as raised, rounded blisters, fluid- or air-filled pockets"
    " underneath paint, coating, or surface layers due to heat, moisture, or"
    " loss of adhesion."
)
SAVED_MODEL_DIR = "./final_bubbling_vlm_lora"

gemini_client = genai.Client()
TEACHER_RESULTS_FILE = Path("teacher_results.json")
STUDENT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

min_pixels = 256 * 28 * 28
max_pixels = 256 * 28 * 28

student_processor = Qwen2VLProcessor.from_pretrained(
    STUDENT_MODEL_ID,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)

training_args = TrainingArguments(
    output_dir="./vlm_bubbling_defect_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    num_train_epochs=3,
    bf16=True,
    gradient_checkpointing=True,
    save_strategy="epoch",
    report_to="none",
    optim="adamw_torch_fused",
    remove_unused_columns=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)


# ---------------------------------------------------------------------------
# 2. Pydantic Models
# ---------------------------------------------------------------------------
class DefectPrediction(BaseModel):
    bubbling: bool = Field(
        description="Whether bubbling surface defect is present"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(description="Brief physical description of evidence")


class StudentDefectResponse(BaseModel):
    bubbling: bool = Field(description="True if bubbling is present, False otherwise")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score")

    @field_validator("bubbling", mode="before")
    def parse_bubbling_bool(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            clean_val = value.strip().upper()
            if clean_val in ["YES", "TRUE", "1"]:
                return True
            if clean_val in ["NO", "FALSE", "0"]:
                return False
        raise ValueError(f"Unable to parse '{value}' into boolean.")

    @field_validator("confidence", mode="before")
    def normalize_confidence(cls, value):
        if value is None:
            return 0.5
        val = float(value)
        if val > 1.0:
            val /= 100.0
        return max(0.0, min(1.0, val))


# ---------------------------------------------------------------------------
# 3. Model Loaders & Helpers
# ---------------------------------------------------------------------------
def initialize_fresh_student_model():
    """Instantiates base student model and wraps with LoRA adapters."""
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )

    model = get_peft_model(qwen_model, lora_config)
    qwen_model.gradient_checkpointing_enable()
    if hasattr(qwen_model, "enable_input_require_grads"):
        qwen_model.enable_input_require_grads()
    return model


def load_fine_tuned_student_model(save_directory: str):
    """Loads fine-tuned LoRA weights onto base Qwen2-VL model."""
    print(f"Loading fine-tuned checkpoint from: {save_directory}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, save_directory)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 4. Data Collator & Dataset
# ---------------------------------------------------------------------------
class Qwen2VLDataCollator:

    def __init__(self, processor):
        self.processor = processor
        self.image_token_ids = [
            self.processor.tokenizer.convert_tokens_to_ids(tok)
            for tok in ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
            if tok in self.processor.tokenizer.get_vocab()
        ]
        self.assistant_header = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def __call__(self, features):
        batch_input_ids = [f["input_ids"] for f in features]
        batch_pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)
        batch_image_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0)

        padded_inputs = self.processor.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding=True,
            return_tensors="pt",
        )

        labels = padded_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        for img_id in self.image_token_ids:
            labels[labels == img_id] = -100

        for i in range(len(labels)):
            seq = padded_inputs["input_ids"][i].tolist()
            for idx in range(len(seq) - len(self.assistant_header)):
                if seq[idx : idx + len(self.assistant_header)] == self.assistant_header:
                    response_start = idx + len(self.assistant_header)
                    labels[i, :response_start] = -100
                    break

        batch_dict = {
            "input_ids": padded_inputs["input_ids"],
            "attention_mask": padded_inputs["attention_mask"],
            "labels": labels,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw,
        }

        if "mm_token_type_ids" in features[0]:
            batch_mm_types = [f["mm_token_type_ids"] for f in features]
            padded_mm_types = self.processor.tokenizer.pad(
                {"input_ids": batch_mm_types},
                padding=True,
                return_tensors="pt",
            )["input_ids"]
            batch_dict["mm_token_type_ids"] = padded_mm_types

        return batch_dict


class VLMDataset(TorchDataset):

    def __init__(self, raw_records, processor):
        self.records = raw_records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        messages = item["messages"]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            images=image, text=[prompt], return_tensors="pt", padding=False
        )

        labels = inputs["input_ids"].clone()

        item_dict = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": labels.squeeze(0),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }

        if "mm_token_type_ids" in inputs:
            item_dict["mm_token_type_ids"] = inputs["mm_token_type_ids"].squeeze(0)

        return item_dict


def extract_metadata(img_path: str) -> str:
    filename = os.path.basename(img_path)
    name_without_ext = os.path.splitext(filename)[0]

    clean_str = re.sub(r"-aID-.*$", "", name_without_ext)
    clean_str = clean_str.replace("_", " ").replace("-", " ")

    words = []
    for chunk in clean_str.split():
        words.extend(wordninja.split(chunk))

    cleaned_text = " ".join(words)
    return f"surface photo showing {cleaned_text}"


def create_formatted_train_dataset(image_paths: list, labels: list, processor) -> VLMDataset:
    raw_records = []
    for img_path, label in zip(image_paths, labels):
        label_text = "YES" if label == 1 else "NO"
        context = extract_metadata(img_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {
                        "type": "text",
                        "text": (
                            f"Context: {context}. Analyze this surface for"
                            f" '{CONCEPT_NAME}' defect. Is the defect present?"
                            " Answer in JSON: {'bubbling': 'YES/NO', 'confidence': 0.0 to 1.0}."
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": f'{{"bubbling": "{label_text}", "confidence": 1.0}}',
            },
        ]
        raw_records.append({"image_path": img_path, "messages": messages})

    return VLMDataset(raw_records, processor)


# ---------------------------------------------------------------------------
# 5. Teacher VLM (Gemini) Routine
# ---------------------------------------------------------------------------
def get_or_query_teacher(image_path: str) -> dict:
    path_obj = Path(image_path)
    img_key = path_obj.name
    results = {}

    if TEACHER_RESULTS_FILE.exists():
        try:
            with open(TEACHER_RESULTS_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = {}

    if img_key in results:
        return results[img_key]

    prompt = f"""
    Analyze this image for the visual defect concept: '{CONCEPT_NAME}'.
    Definition: {CONCEPT_DEF}

    Determine if this exact defect is present in the image.
    Provide your confidence level (0.0 to 1.0) and brief reasoning.
    """

    try:
        image = Image.open(image_path)
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DefectPrediction,
                temperature=0.1,
            ),
        )
        parsed_result: DefectPrediction = response.parsed
        results[img_key] = parsed_result.model_dump()

        with open(TEACHER_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results[img_key]
    except Exception as e:
        print(f"\nTeacher query error on {image_path}: {e}")
        return {"bubbling": False, "confidence": 0.0, "reasoning": "API Error"}


# ---------------------------------------------------------------------------
# 6. Student Fine-tuning, Prediction & Evaluation
# ---------------------------------------------------------------------------
def fine_tune_student(model, formatted_train_dataset, save_path=SAVED_MODEL_DIR):
    """Fine-tunes the model and explicitly saves LoRA weights + processor."""
    data_collator = Qwen2VLDataCollator(student_processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    print(f"\nSaving fine-tuned model and processor to {save_path}...")
    model.save_pretrained(save_path)
    student_processor.save_pretrained(save_path)


def predict_single_image(model, image_path: str) -> tuple[float, int]:
    """Runs inference and parses response with Pydantic."""
    print(".", end="", flush=True)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    context = extract_metadata(image_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {
                "type": "text",
                "text": (
                    f"Context: {context}. Analyze this surface for"
                    f" '{CONCEPT_NAME}' defect. Is the defect present? Answer"
                    ' in JSON: {"bubbling": "YES/NO", "confidence": 0.0 to 1.0}.'
                ),
            },
        ],
    }]

    prompt = student_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = student_processor(
        images=image, text=[prompt], return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = student_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0].strip()

    # Attempt parsing with Pydantic
    try:
        json_match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if json_match:
            raw_json = json.loads(json_match.group(0))
            parsed = StudentDefectResponse.model_validate(raw_json)
            return parsed.confidence, 1 if parsed.bubbling else 0
    except Exception:
        pass

    # Robust regex fallback if JSON structure fails
    cleaned_text = response_text.upper()
    if "YES" in cleaned_text:
        return 0.85, 1
    if "NO" in cleaned_text:
        return 0.15, 0

    return 0.50, 0


def evaluate_student(model, eval_paths, eval_gt) -> dict:
    """Evaluates student predictions against ground truth labels."""
    y_true, y_pred = [], []

    for img_path, gt in zip(eval_paths, eval_gt):
        _, pred = predict_single_image(model, img_path)
        y_pred.append(pred)
        y_true.append(gt)

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"precision": p, "recall": r, "f1": f1}


def construct_dataset(pos_dir: str, neg_dir: str, category: str) -> tuple[list, list]:
    pos_images = glob.glob(os.path.join(pos_dir, "*.*"))
    neg_images = glob.glob(os.path.join(neg_dir, "*.*"))

    dataset_paths = []
    for img_path in pos_images:
        if os.path.basename(img_path).startswith(category):
            dataset_paths.append(img_path)
            for rotated_img_path in pos_images:
                if os.path.basename(rotated_img_path).startswith("rotated_") and os.path.basename(rotated_img_path).endswith(os.path.basename(img_path).replace(category, "")):
                    dataset_paths.append(rotated_img_path)
    dataset_gt = [1] * len(dataset_paths)
    for img_path in neg_images:
        if os.path.basename(img_path).startswith(category):
            dataset_paths.append(img_path)
            for rotated_img_path in neg_images:
                if os.path.basename(rotated_img_path).startswith("rotated_") and os.path.basename(rotated_img_path).endswith(os.path.basename(img_path).replace(category, "")):
                    dataset_paths.append(rotated_img_path)
  
    dataset_gt += [0] * (len(dataset_paths) - len(dataset_gt))
    return dataset_paths, dataset_gt


# ---------------------------------------------------------------------------
# 7. Execution Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(pos_dir: str, neg_dir: str):
    print("\n--- Initializing Base Student Model ---")
    student_model = initialize_fresh_student_model()

    print("\n--- Loading & Splitting Data ---")
    pos_images = glob.glob(os.path.join(pos_dir, "*.*"))
    neg_images = glob.glob(os.path.join(neg_dir, "*.*"))
    print(f"Positive images: {len(pos_images)} | Negative images: {len(neg_images)}")
    
    eval_paths, eval_gt = construct_dataset(pos_dir, neg_dir, "eval_")
    test_paths, test_gt = construct_dataset(pos_dir, neg_dir, "test_")

    training_dataset = [image_path for image_path in pos_images if image_path not in eval_paths + test_paths]
    training_data_gt = [1] * len(training_dataset)
    training_dataset += [image_path for image_path in neg_images if image_path not in eval_paths + test_paths]
    training_data_gt += [0] * (len(training_dataset) - len(training_data_gt))

    print(
        f"Eval Set Size: {len(eval_paths)} ({sum(eval_gt)} pos, {len(eval_gt) - sum(eval_gt)} neg) | Test Set Size:"
        f" {len(test_paths)} ({sum(test_gt)} pos, {len(test_gt) - sum(test_gt)} neg) | Training Set Size:"
        f" {len(training_dataset)} ({sum(training_data_gt)} pos,"
        f" {len(training_data_gt) - sum(training_data_gt)} neg)"
    )

    print("\n--- Zero-Shot Baseline Evaluation ---")
    baseline_metrics = evaluate_student(student_model, eval_paths, eval_gt)
    print(
        f"\nBaseline -> Precision: {baseline_metrics['precision']:.2f}, Recall:"
        f" {baseline_metrics['recall']:.2f}, F1: {baseline_metrics['f1']:.2f}"
    )

    print("\n--- Teacher Supervision Loop ---")
    CONFIDENCE_THRESHOLD = 0.75
    train_image_paths, train_labels = [], []

    for idx, img_path in enumerate(training_dataset):
        print(".", end="", flush=True)

        if training_data_gt[idx] == -1:
            teacher_res = get_or_query_teacher(img_path)
        else:
            teacher_res = {
                "bubbling": training_data_gt[idx] == 1,
                "confidence": 1.0,
                "reasoning": "Ground truth label",
            }

        if teacher_res["confidence"] >= CONFIDENCE_THRESHOLD:
            train_image_paths.append(img_path)
            train_labels.append(1 if teacher_res["bubbling"] else 0)

    print(f"\nConstructed Training Set: {len(train_image_paths)} pseudo-labeled records.")
    formatted_train_dataset = create_formatted_train_dataset(
        train_image_paths, train_labels, student_processor
    )

    print("\n--- Training Student Model (Qwen2-VL LoRA) ---")
    fine_tune_student(student_model, formatted_train_dataset, save_path=SAVED_MODEL_DIR)

    # Completely unload model from GPU to ensure clean evaluate reload
    print("\n--- Unloading training model and clearing CUDA memory ---")
    del student_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- Loading Fine-Tuned Model Checkpoint Before Evaluation ---")
    eval_student_model = load_fine_tuned_student_model(SAVED_MODEL_DIR)

    print("\n--- Post-Adaptation Evaluation ---")
    post_metrics = evaluate_student(eval_student_model, eval_paths, eval_gt)
    print(
        f"\nPost-Train -> Precision: {post_metrics['precision']:.2f}, Recall:"
        f" {post_metrics['recall']:.2f}, F1: {post_metrics['f1']:.2f}"
    )

    print("\n--- Active Learning Loop (Uncertainty Sampling Demo) ---")
    for img_path in test_paths:
        prob, pred = predict_single_image(eval_student_model, img_path)
        print(
            f"\nImage: {os.path.basename(img_path)} | Student Prob: {prob:.3f} | Pred: {pred}"
        )

        if 0.35 <= prob <= 0.65:
            print("  --> [Routed to Teacher] Student uncertain. Re-querying teacher VLM...")

def test_ood_images():
    print("\n--- Out-of-Distribution (OOD) Image Evaluation ---")
    eval_student_model = load_fine_tuned_student_model(SAVED_MODEL_DIR)
    ood_image_paths = glob.glob("./OOD data/*.*")

    for img_path in ood_image_paths:
        prob, pred = predict_single_image(eval_student_model, img_path)
        print(
            f"\nImage: {os.path.basename(img_path)} | Student Prob: {prob:.3f} | Pred: {pred}"
        )

if __name__ == "__main__":
    POS_DIR = "./train-bubbling"
    NEG_DIR = "./hard-negatives-bubbling"

    run_pipeline(POS_DIR, NEG_DIR)

    test_ood_images()