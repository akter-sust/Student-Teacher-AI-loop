import os

BASE_CACHE = "/mnt/data/cache"
# Set Hugging Face, Triton, and PyTorch environment variables
os.environ["HF_HOME"] = f"{BASE_CACHE}/huggingface"
os.environ["HF_HUB_CACHE"] = f"{BASE_CACHE}/huggingface/hub"
os.environ["TMPDIR"] = f"{BASE_CACHE}/tmp"
os.environ["TRITON_CACHE_DIR"] = f"{BASE_CACHE}/triton"
os.environ["TORCH_HOME"] = f"{BASE_CACHE}/torch"

# Create all directories explicitly
for path in ["huggingface/hub", "tmp", "triton", "torch"]:
  os.makedirs(f"{BASE_CACHE}/{path}", exist_ok=True)

# Disable PyTorch native Triton router fallback
os.environ["TORCH_NATIVE_TRITON_SEARCH_PATH"] = "0"
os.environ["TRITON_DISABLE"] = "1"


import glob
import json
from pathlib import Path
import re
from PIL import Image
from google import genai
from google.genai import types
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel, Field
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
# 1. Configuration & Initializations
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONCEPT_NAME = "bubbling"
CONCEPT_DEF = (
    "Bubbling appears as raised, rounded blisters, fluid- or air-filled pockets"
    " underneath paint, coating, or surface layers due to heat, moisture, or"
    " loss of adhesion."
)

# Initialize Gemini Client for Teacher VLM
gemini_client = genai.Client()  # Requires GEMINI_API_KEY set in environment
TEACHER_RESULTS_FILE = Path("teacher_results.json")

# Load Qwen2-VL Base Model and Processor for Student
student_model_id = "Qwen/Qwen2-VL-2B-Instruct"

# Restrict image dimensions (e.g., max ~768x768 resolution)
# 28x28 is the base patch size for Qwen2-VL
min_pixels = 256 * 28 * 28
max_pixels = 768 * 28 * 28

student_processor = Qwen2VLProcessor.from_pretrained(
    student_model_id,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)

qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    student_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Configure LoRA Adapters for VLM
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

student_model = get_peft_model(qwen_model, lora_config)
student_model.print_trainable_parameters()

qwen_model.gradient_checkpointing_enable()
if hasattr(qwen_model, "enable_input_require_grads"):
  qwen_model.enable_input_require_grads()


# Training Arguments for Hugging Face Trainer
training_args = TrainingArguments(
    output_dir="./vlm_bubbling_defect_lora",
    per_device_train_batch_size=1,  # Keep batch size = 1 for dynamic vision resolution
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=1,
    num_train_epochs=10,
    bf16=True,  # Match bfloat16 loading dtype
    gradient_checkpointing=True,
    save_strategy="epoch",
    report_to="none",
    optim="adamw_torch_fused",
    remove_unused_columns=False,  # CRITICAL: Do not drop image_grid_thw or pixel_values
)


# ---------------------------------------------------------------------------
# 2. Custom Data Collator for Qwen2-VL
# ---------------------------------------------------------------------------
class Qwen2VLDataCollator:

  def __init__(self, processor):
    self.processor = processor

  def __call__(self, features):
    batch_input_ids = [f["input_ids"] for f in features]
    batch_labels = [f["labels"] for f in features]
    batch_pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)
    batch_image_grid_thw = torch.cat(
        [f["image_grid_thw"] for f in features], dim=0
    )

    padded_inputs = self.processor.tokenizer.pad(
        {"input_ids": batch_input_ids},
        padding=True,
        return_tensors="pt",
    )

    padded_labels = self.processor.tokenizer.pad(
        {"input_ids": batch_labels},
        padding=True,
        return_tensors="pt",
    )["input_ids"]

    padded_labels[
        padded_labels == self.processor.tokenizer.pad_token_id
    ] = -100

    batch_dict = {
        "input_ids": padded_inputs["input_ids"],
        "attention_mask": padded_inputs["attention_mask"],
        "labels": padded_labels,
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

# ---------------------------------------------------------------------------
# 3. Data Structures & Helper Functions
# ---------------------------------------------------------------------------
class DefectPrediction(BaseModel):
  bubbling: bool = Field(
      description="Whether bubbling surface defect is present"
  )
  confidence: float = Field(
      description="Confidence score between 0.0 and 1.0"
  )
  reasoning: str = Field(description="Brief physical description of evidence")


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
  """Splits concatenated filenames and cleans metadata descriptions."""
  filename = os.path.basename(img_path)
  name_without_ext = os.path.splitext(filename)[0]

  clean_str = re.sub(r"-aID-.*$", "", name_without_ext)
  clean_str = clean_str.replace("_", " ").replace("-", " ")

  words = []
  for chunk in clean_str.split():
    words.extend(wordninja.split(chunk))

  cleaned_text = " ".join(words)
  return f"surface photo showing {cleaned_text}"


def create_formatted_train_dataset(
    image_paths: list, labels: list, processor
) -> VLMDataset:
  """Formats training data records into instruction prompts for VLM fine-tuning."""
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
    raw_records.append({
        "image_path": img_path,
        "messages": messages,
    })

  return VLMDataset(raw_records, processor)


# ---------------------------------------------------------------------------
# 4. Teacher VLM (Gemini) Query Routine
# ---------------------------------------------------------------------------
def get_or_query_teacher(image_path: str) -> dict:
  """Queries Gemini 2.5 Flash with structured Pydantic response schema."""
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
        model="gemini-2.5-flash",
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
# 5. Student VLM Training & Evaluation
# ---------------------------------------------------------------------------
def fine_tune_student(student_model, formatted_train_dataset):
  """Fine-tunes the Qwen2-VL model using LoRA adapters."""
  data_collator = Qwen2VLDataCollator(student_processor)

  trainer = Trainer(
      model=student_model,
      args=training_args,
      train_dataset=formatted_train_dataset,
      data_collator=data_collator,
  )

  trainer.train()
  student_model.save_pretrained("./final_bubbling_vlm_lora")
  student_processor.save_pretrained("./final_bubbling_vlm_lora")
  return student_model

def predict_single_image(model, image_path: str) -> tuple[float, int]:
  """Runs inference on a single image and returns predicted defect probability and binary class."""
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

  # Default fallback values if parsing fails
  confidence = 0.50
  label = 0

  # 1. Attempt structured JSON parsing
  try:
    # Match JSON object inside text, ignoring potential markdown code fences
    print(f"\nRaw Student Response: {response_text}")
    json_match = re.search(r"\{.*?\}", response_text, re.DOTALL)
    if json_match:
      data = json.loads(json_match.group(0))
      print(f"Extracted JSON: {data}")

      # Extract 'bubbling' status (accepting string or bool)
      bubbling_val = str(data.get("bubbling", "")).strip().upper()
      if bubbling_val in ["YES", "TRUE", "1"]:
        label = 1
      elif bubbling_val in ["NO", "FALSE", "0"]:
        label = 0

      # Extract and validate confidence float
      raw_conf = data.get("confidence", None)
      if raw_conf is not None:
        confidence = float(raw_conf)
        # Convert scale if model outputs percentage (e.g. 85 instead of 0.85)
        if confidence > 1.0:
          confidence /= 100.0
        confidence = max(0.0, min(1.0, confidence))

      return confidence, label

  except (json.JSONDecodeError, ValueError, TypeError):
    pass  # Fall back to text parsing below if JSON extraction fails

  # 2. Robust fallback if JSON parsing fails
  cleaned_text = response_text.upper()
  if "YES" in cleaned_text:
    label = 1
    confidence = 0.85
  elif "NO" in cleaned_text:
    label = 0
    confidence = 0.15

  return confidence, label


def evaluate_student(student_model, eval_paths, eval_gt) -> dict:
  """Evaluates Student VLM predictions against ground truth validation split."""
  y_true, y_pred = [], []

  for img_path, gt in zip(eval_paths, eval_gt):
    _, pred = predict_single_image(student_model, img_path)
    y_pred.append(pred)
    y_true.append(gt)

  p = precision_score(y_true, y_pred, zero_division=0)
  r = recall_score(y_true, y_pred, zero_division=0)
  f1 = f1_score(y_true, y_pred, zero_division=0)
  return {"precision": p, "recall": r, "f1": f1}


# ---------------------------------------------------------------------------
# 6. Full Pipeline Execution
# ---------------------------------------------------------------------------
def run_pipeline(pos_dir: str, neg_dir: str):
  """Executes data splitting, zero-shot baselining, teacher supervision, student fine-tuning, and active learning routing."""
  print("\n--- Loading & Splitting Data ---")
  pos_images = glob.glob(os.path.join(pos_dir, "*.*"))
  neg_images = glob.glob(os.path.join(neg_dir, "*.*"))
  print(
      f"Positive images: {len(pos_images)} | Negative images: {len(neg_images)}"
  )

  # Splits: 5 pos / 3 neg for Eval set, 2 pos / 1 neg for Test set, rest for Training
  eval_paths = pos_images[:5] + neg_images[:3]
  eval_gt = [1, 1, 1, 1, 1, 0, 0, 0]

  test_set = pos_images[5:7] + neg_images[3:4]

  training_dataset = pos_images[7:] + neg_images[4:]
  training_data_gt = [1] * len(pos_images[7:]) + [0] * len(neg_images[4:])

  print(
      f"Eval Set Size: {len(eval_paths)} (5 pos, 3 neg) | Test Set Size:"
      f" {len(test_set)} (2 pos, 1 neg) | Training Set Size:"
      f" {len(training_dataset)} ({len(pos_images)-7} pos,"
      f" {len(neg_images)-4} neg)"
  )

  print("\n--- Zero-Shot Baseline Evaluation ---")
  baseline_metrics = evaluate_student(student_model, eval_paths, eval_gt)
  print(
      f"Baseline -> Precision: {baseline_metrics['precision']:.2f}, Recall:"
      f" {baseline_metrics['recall']:.2f}, F1: {baseline_metrics['f1']:.2f}"
  )

  print("\n--- Teacher Supervision Loop ---")
  CONFIDENCE_THRESHOLD = 0.75
  train_image_paths, train_labels = [], []

  print("Gathering teacher annotations for training set...")
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

  print(
      f"\nConstructed Training Set: {len(train_image_paths)} pseudo-labeled"
      " records."
  )
  formatted_train_dataset = create_formatted_train_dataset(
      train_image_paths, train_labels, student_processor
  )

  print("\n--- Training Student Model (Qwen2-VL LoRA) ---")
  fine_tune_student(student_model, formatted_train_dataset)

  print("\n--- Post-Adaptation Evaluation ---")
  post_metrics = evaluate_student(student_model, eval_paths, eval_gt)
  print(
      f"Post-Train -> Precision: {post_metrics['precision']:.2f}, Recall:"
      f" {post_metrics['recall']:.2f}, F1: {post_metrics['f1']:.2f}"
  )

  print("\n--- Active Learning Loop (Uncertainty Sampling Demo) ---")
  for img_path in test_set:
    prob, pred = predict_single_image(student_model, img_path)
    print(
        f"Image: {os.path.basename(img_path)} | Student Prob: {prob:.3f} | Pred:"
        f" {pred}"
    )

    # Uncertainty Routing Band (0.35 <= P <= 0.65)
    if 0.35 <= prob <= 0.65:
      print(
          "  --> [Routed to Teacher] Student uncertain. Re-querying teacher"
          " VLM..."
      )
    #   teacher_res = get_or_query_teacher(img_path)
    #   if teacher_res["confidence"] >= CONFIDENCE_THRESHOLD:
    #     print(
    #         f"      Teacher Label: {teacher_res['bubbling']} | Conf:"
    #         f" {teacher_res['confidence']} -> Adding to next iteration buffer."
    #     )
    #    else:
    #     print(
    #         f"      Teacher Label: {teacher_res['bubbling']} | Conf:"
    #         f" {teacher_res['confidence']} -> Discarding low-confidence label. Required HITL"
    #     )


if __name__ == "__main__":
  POS_DIR = "./train-bubbling"
  NEG_DIR = "./hard-negatives-bubbling"

  run_pipeline(POS_DIR, NEG_DIR)