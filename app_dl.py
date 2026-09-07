from pathlib import Path
import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import CLIPProcessor, CLIPModel
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset
import re
import wordninja

# ---------------------------------------------------------------------------
# 1. Configuration & Initializations
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONCEPT_NAME = "bubbling"
CONCEPT_DEF = (
    "Bubbling appears as raised, rounded blisters, fluid- or air-filled pockets "
    "underneath paint, coating, or surface layers due to heat, moisture, or loss of adhesion."
)

# Initialize CLIP for Student feature extraction
print("Loading vision encoder (CLIP)...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize Gemini Client for Teacher
gemini_client = genai.Client() # set GEMINI_API_KEY in your environment
TEACHER_RESULTS_FILE = Path("teacher_results.json")


# ---------------------------------------------------------------------------
# 2. Student Model Architecture (Lightweight Classifier Head)
# ---------------------------------------------------------------------------
class StudentClassifier(nn.Module):
    def __init__(
        self,
        img_dim=512,
        text_dim=512,
        hidden_dim=256,
        metadata_scale=1.5,
    ):
        super().__init__()
        # Explicit scale factor applied to text features
        self.metadata_scale = metadata_scale

        # Input layers
        self.fc_img = nn.Linear(img_dim, 128)
        self.fc_text = nn.Linear(text_dim, 128)

        # Combined decision network
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # Split input back into image (first 512) and text (last 512)
        img_x = x[:, :512]
        text_x = x[:, 512:] * self.metadata_scale

        # Project sub-vectors
        img_feat = torch.relu(self.fc_img(img_x))
        text_feat = torch.relu(self.fc_text(text_x))

        # Combine projected features (128 + 128 = 256)
        combined = torch.cat([img_feat, text_feat], dim=-1)

        return self.classifier(combined)

class EmbeddingDataset(Dataset):
  """Dataset wrapper for pre-computed 1D feature embeddings and labels."""

  def __init__(self, data_list):
    # Expects a list of dicts: [{'embedding': tensor, 'label': float/bool}]
    self.embeddings = torch.stack([item['embedding'] for item in data_list])
    self.labels = torch.tensor(
        [float(item['label']) for item in data_list], dtype=torch.float32
    )

  def __len__(self):
    return len(self.embeddings)

  def __getitem__(self, idx):
    return self.embeddings[idx], self.labels[idx]

class DefectPrediction(BaseModel):
    bubbling: bool = Field(
        description="Whether bubbling surface defect is present"
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief physical description of evidence")

def extract_text_features(text):
  """Encodes cleaned text prompt into a 512-dim CLIP text vector."""
  inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
  with torch.no_grad():
    text_features = clip_model.get_text_features(**inputs)

  # Normalize L2 norm
  text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
  return text_features.squeeze(0)

def extract_image_features(image_path: str) -> torch.Tensor:
    """Extracts a 512-dim embedding from an image using frozen CLIP encoder."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    # Normalize embeddings
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze(0)

def extract_metadata(img_path: str) -> str:
  """Splits concatenated filenames and handles spelling mistakes dynamically.

  Example: 'Bubbling_pantedplasterboardceilngother' -> 'Bubbling panted
  plaster board ceilng other'
  """
  filename = os.path.basename(img_path)
  name_without_ext = os.path.splitext(filename)[0]

  # Remove hash IDs if present
  clean_str = re.sub(r"-aID-.*$", "", name_without_ext)

  # Replace underscores/hyphens with spaces
  clean_str = clean_str.replace("_", " ").replace("-", " ")

  # Split jammed words dynamically using wordninja (handles typos naturally)
  words = []
  for chunk in clean_str.split():
    # wordninja splits "pantedplasterboard" into ["panted", "plaster", "board"]
    split_words = wordninja.split(chunk)
    words.extend(split_words)

  cleaned_text = " ".join(words)

  return f"surface photo showing {cleaned_text}"

def extract_combined_features(img_path):
  """Generates a 1024-dim vector: 512-dim visual vector + 512-dim text metadata vector."""
  # Image embedding (512-dim)
  visual_vec = extract_image_features(img_path)

  # Text metadata embedding (512-dim)
  metadata_text = extract_metadata(img_path)
  text_vec = extract_text_features(metadata_text)

  # Concatenate into single 1024-dim vector
  combined_x = torch.cat([visual_vec, text_vec], dim=-1)

  return combined_x

def save_student_checkpoint(model, optimizer, path="student_pretrained.pth"):
  """Saves model weights and optimizer state to disk."""
  checkpoint = {
      "model_state": model.state_dict(),
      "optimizer_state": optimizer.state_dict(),
  }
  torch.save(checkpoint, path)
  print(f"Saved student model checkpoint to: {path}")

def load_student_checkpoint(
    model, optimizer=None, path="student_pretrained.pth", lr=1e-3
):
  """Loads checkpoint if it exists; otherwise starts with fresh weights."""
  if os.path.exists(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
      optimizer.load_state_dict(checkpoint["optimizer_state"])
    lr = 1e-6
    print(f"Loaded pretrained weights from: {path}")
  else:
    print(f"No checkpoint found at '{path}'. Starting fresh training...")

  return model, optimizer, lr

def get_or_query_teacher(image_path: str) -> dict:
    """Queries Gemini VLM with structured prompt and JSON response schema."""
    path_obj = Path(image_path)
    img_key = path_obj.name
    results = {}
    if TEACHER_RESULTS_FILE.exists():
        try:
            with open(TEACHER_RESULTS_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = {}

    # Return saved output if already in file
    if img_key in results:
        # print(f"[Loaded from JSON] Skipping API call for: {img_key}")
        return results[img_key]

    prompt = f"""
    Analyze this image for the visual defect concept: '{CONCEPT_NAME}'.
    Definition: {CONCEPT_DEF}

    Determine if this exact defect is present in the image.
    Provide your confidence level (0.0 to 1.0) and brief reasoning.
    """

    try:
        image = Image.open(image_path)
        chat = gemini_client.chats.create(
            model="gemini-3.5-flash",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DefectPrediction,
                temperature=0.1,
            ),
        )
        response = chat.send_message([image, prompt])
        parsed_result: DefectPrediction = response.parsed
        results[img_key] = parsed_result.model_dump() 

        with open(TEACHER_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results[img_key]
    except Exception as e:
        print(f"Teacher query error on {image_path}: {e}")
        return {"bubbling": False, "confidence": 0.0, "reasoning": "API Error"}

def fine_tune_student(
    student_model,
    new_train_data,
    checkpoint_path='student_pretrained.pth',
    epochs=30,
    lr=1e-3,
):
  """Loads a pretrained student model and fine-tunes it on new embedding data."""
  if not new_train_data:
    print('No new data provided for fine-tuning.')
    return student_model

  student_model = student_model.to(DEVICE)
  optimizer = optim.Adam(student_model.parameters(), lr=lr)
  # for imblance data, if data is not imbalance don't pass pos_weight
  pos_weight = torch.tensor([1.0]).to(DEVICE)
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

  # Load pretrained state
  student_model, optimizer, lr = load_student_checkpoint(
      student_model, optimizer, checkpoint_path, lr=lr
  )

  # Create dataset & loader from new cached embeddings
  dataset = EmbeddingDataset(new_train_data)
  loader = DataLoader(dataset, batch_size=32, shuffle=True)

  student_model.train()
  # Freeze BatchNorm parameters during fine-tuning to avoid probability collapse
  for m in student_model.modules():
    if isinstance(m, nn.BatchNorm1d):
      m.eval()

  patience = 3  # Stop if loss doesn't improve for 3 consecutive epochs
  min_delta = 0.001  # Minimum loss change to count as improvement
  best_loss = float("inf")
  no_improve_count = 0

  for epoch in range(epochs):
    epoch_loss = 0.0
    for embeddings, labels in loader:
      embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)
      optimizer.zero_grad()
      preds = student_model(embeddings).squeeze(-1)
      loss = criterion(preds, labels)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Check improvement
    if best_loss - avg_loss > min_delta:
      best_loss = avg_loss
      no_improve_count = 0
    else:
      no_improve_count += 1

    if no_improve_count >= patience:
      print(
          f"Loss plateaued around {avg_loss:.4f} for {patience} epochs. Stopping."
      )
      break

  # Save updated checkpoint
  save_student_checkpoint(student_model, optimizer, checkpoint_path)
  return student_model

def evaluate_student(student_model, eval_data):
    """Evaluates student performance on a ground-truth holdout set."""
    student_model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for item in eval_data:
            emb = item['embedding'].unsqueeze(0).to(DEVICE)
            prob = student_model(emb).item()
            y_pred.append(1 if prob >= 0.5 else 0)
            y_true.append(item['ground_truth'])

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"precision": p, "recall": r, "f1": f1}

def run_pipeline(pos_dir: str, neg_dir: str):
    """Executes the complete student-teacher feedback loop."""
    print("\n--- Loading & Splitting Data ---")
    pos_images = glob.glob(os.path.join(pos_dir, "*.*"))
    neg_images = glob.glob(os.path.join(neg_dir, "*.*"))
    print(f"Positive images: {len(pos_images)} | Negative images: {len(neg_images)}")

    # Split: 3 pos / 3 neg for Eval set, 1 pos / 1 neg for test set, remaining for Unlabeled Pool
    eval_paths = pos_images[:5] + neg_images[:3]
    eval_gt = [1, 1, 1, 1, 1, 0, 0, 0]
    test_set = pos_images[5:7] + neg_images[3:4]
    training_dataset = pos_images[7:] + neg_images[4:]
    training_data_gt = [1]*len(pos_images[7:]) + [0]*len(neg_images[4:])
    print(f"Eval Set Size: {len(eval_paths)} (5 pos, 3 neg) | Test Set Size: {len(test_set)} (2 pos, 1 neg) | Training Set Size: {len(training_dataset)} ({len(pos_images)-7} pos, {len(neg_images)-4} neg)")

    print("Extracting features for eval set")
    eval_data = [
        {"embedding": extract_combined_features(p), "ground_truth": gt}
        for p, gt in zip(eval_paths, eval_gt)
    ]

    student = StudentClassifier().to(DEVICE)
    train_dataset = []

    print("\n--- Zero-Shot Baseline Evaluation ---")
    baseline_metrics = evaluate_student(student, eval_data)
    print(f"Baseline -> Precision: {baseline_metrics['precision']:.2f}, Recall: {baseline_metrics['recall']:.2f}, F1: {baseline_metrics['f1']:.2f}")

    print("\n--- Teacher Supervision Loop ---")
    CONFIDENCE_THRESHOLD = 0.75  # Filter out uncertain teacher output

    print("Getting teacher labels for training set")
    for idx, img_path in enumerate(training_dataset):
        
        print(".", end="")
        # print(f"\nProcessing Image {idx+1}/{len(training_dataset)}: {os.path.basename(img_path)}")
        if training_data_gt[idx] == -1:
            teacher_res = get_or_query_teacher(img_path)
        else:
            teacher_res = {
                "bubbling": training_data_gt[idx],
                "confidence": 1.0,
                "reasoning": "Ground truth label"
            }
        # print(f"  Teacher Label: {teacher_res['bubbling']} | Conf: {teacher_res['confidence']} | Reason: {teacher_res['reasoning']}")

        if teacher_res['confidence'] >= CONFIDENCE_THRESHOLD:
            feat = extract_combined_features(img_path)
            train_dataset.append({
                "embedding": feat,
                "label": 1.0 if teacher_res['bubbling'] else 0.0
            })
        else:
            print("  [Flagged] Teacher confidence below threshold. Skipping or routing to HITL.")

    print(f"\nConstructed Training Set: {len(train_dataset)} pseudo-labeled records.")

    print("\n--- Training Student Model ---")
    student = fine_tune_student(student, train_dataset)

    print("\n--- Post-Adaptation Evaluation ---")
    post_metrics = evaluate_student(student, eval_data)
    print(f"Post-Train -> Precision: {post_metrics['precision']:.2f}, Recall: {post_metrics['recall']:.2f}, F1: {post_metrics['f1']:.2f}")

    print("\n--- Active Learning Loop (Uncertainty Sampling Demo) ---")
    # Simulate scanning a new batch to route low-confidence student predictions
    student.eval()
    with torch.no_grad():
        for img_path in test_set:
            emb = extract_combined_features(img_path).unsqueeze(0).to(DEVICE)
            logits = student(emb)
            prob = torch.sigmoid(logits).item()
            print(f"Image: {os.path.basename(img_path)} | Student Prob: {prob:.3f}")
            
            # Uncertainty band between 0.35 and 0.65
            if 0.35 <= prob <= 0.65:
                print("  --> [Routed to Loop] Student uncertain. Re-querying teacher or human annotator.")
            # teacher_res = get_or_query_teacher(img_path)
            # if teacher_res['confidence'] >= CONFIDENCE_THRESHOLD:
            #     # add it to training dataset for next iteration
            # else:
            #     # Wait for human annotator
            #     # potential noisy data
            #     # if it is good data, add it to training dataset for next iteration
        

    

if __name__ == "__main__":
    # Point these to your image folders
    POS_DIR = "./train-bubbling"
    NEG_DIR = "./hard-negatives-bubbling"

    run_pipeline(POS_DIR, NEG_DIR)