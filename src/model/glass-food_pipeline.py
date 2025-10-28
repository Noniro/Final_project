"""
GLASS-FOOD reproduction patch
--------------------------------
This script implements the GLASS-FOOD pipeline as described in the paper
"Extending limited datasets with GAN-like self-supervision for SMS spam detection."
It consists of three stages that can be run sequentially via command-line arguments.

1) augment: Cosine-similarity based augmentation. For each message (ham and spam),
            it identifies the "farthest" token based on embedding distance, masks it,
            and replaces it to generate a new, paired sample. This matches the
            generator-like behavior described in the paper.

2) train:   Trains a RoBERTa-based discriminator on the augmented dataset. In line
            with the paper, it does not use class-weights. It then determines the
            optimal message-level pseudo-labeling threshold (τ) by maximizing the
            F1-score on a validation set. This threshold is used for OOD detection,
            where the 'spam' probability serves as the OOD score.

3) eval:    Evaluates the trained discriminator and the selected threshold τ on the
            unseen test set, reporting final classification metrics.

Your TextAugmenter implementation from `roberta_aug.py` is a required dependency.
Please ensure the file is in the same directory or accessible via PYTHONPATH.
"""

# =================================================================_
# Imports
# =================================================================_

import os
import json
import math
import pandas as pd
import numpy as np
from typing import Optional

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    IntervalStrategy,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# Attempt to import the user's custom TextAugmenter
try:
    from roberta_aug import TextAugmenter
except ImportError:
    print("Error: Could not import 'TextAugmenter' from 'roberta_aug.py'.")
    print("Please ensure 'roberta_aug.py' is in the same directory or in your PYTHONPATH.")
    exit()


# =================================================================_
# Configuration
# =================================================================_

# --- Directory and File Paths ---
# Adjust these paths if your project structure is different.
PROCESSED_DATA_DIR = "../../data/processed/"
MODEL_DIR = "../../models/glassfood_discriminator/"
LOG_DIR = "../../logs/glassfood_discriminator/"

# --- Input Files (Updated as per user request) ---
TRAIN_INPUT_FILE = "train_sms_dedup.csv"
TEST_INPUT_FILE = "test_sms_dedup.csv"

# --- Generated/Intermediate File ---
AUGMENTED_TRAIN_FILE = "train_sms_glassfood_aug_roberta.csv"

# --- Model and Training Hyperparameters ---
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512
SEED = 42
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


# =================================================================_
# STAGE 1: Augment Data (Generator-like)
# =================================================================_

def augment_all_messages(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Applies cosine-similarity based augmentation to every message in the dataframe.

    This function mimics the "generator" part of the GLASS-FOOD architecture.
    It finds the token most semantically distant from the overall message,
    replaces it, and creates a new dataset containing both the original and
    the augmented messages as pairs.

    Args:
        df: DataFrame with 'message' and 'label' columns.
        seed: Random seed for reproducibility.

    Returns:
        A new DataFrame containing original and augmented data, shuffled.
    """
    df = df.copy()
    assert {"message", "label"}.issubset(df.columns), "Input DataFrame must contain 'message' and 'label' columns."

    print("Initializing TextAugmenter for data generation...")
    augmenter = TextAugmenter()

    texts = df["message"].astype(str).fillna("").tolist()

    # Batch augmentation for efficiency
    batch_size = 64
    augmented_texts = []
    print(f"Starting augmentation in batches of {batch_size}...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        augmented_batch = augmenter.replace_farthest_token_batch(batch)
        augmented_texts.extend(augmented_batch)
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i + len(batch)} of {len(texts)} messages.")

    df_aug = pd.DataFrame({
        "message": augmented_texts,
        "label": df["label"].tolist()
    })

    # The paper's methodology creates a 1:1 pairing. We combine originals and
    # the newly generated texts.
    df_out = pd.concat([df, df_aug], ignore_index=True)

    # Shuffle the combined dataset for training
    df_out = df_out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df_out


def main_augment():
    """CLI entry point for the data augmentation stage."""
    print("--- Running Stage 1: Data Augmentation ---")
    in_path = os.path.join(PROCESSED_DATA_DIR, TRAIN_INPUT_FILE)
    out_path = os.path.join(PROCESSED_DATA_DIR, AUGMENTED_TRAIN_FILE)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing input file: {in_path}. Please check your PROCESSED_DATA_DIR path.")

    df = pd.read_csv(in_path)
    print(f"Loaded original training data from {in_path}:")
    print(f"  Shape: {df.shape}")
    print(f"  Label distribution:\n{df['label'].value_counts(normalize=True)}\n")

    df_augmented = augment_all_messages(df)

    print("\nAugmentation complete. Final dataset stats:")
    print(f"  Shape: {df_augmented.shape}")
    print(f"  Label distribution:\n{df_augmented['label'].value_counts(normalize=True)}\n")

    df_augmented.to_csv(out_path, index=False)
    print(f"Saved augmented dataset to: {out_path}")


# =================================================================_
# STAGE 2: Train Discriminator & Find Threshold (τ)
# =================================================================_

class SMSDataset(torch.utils.data.Dataset):
    """PyTorch Dataset class for SMS messages."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    """Computes F1, precision, recall, and accuracy for binary classification."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1, zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": p, "recall": r}


class PlainTrainer(Trainer):
    """
    Custom Trainer that uses standard CrossEntropyLoss without class weights,
    as specified in the GLASS-FOOD paper.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): # <--- THE CORRECTED LINE
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def fit_and_pick_threshold(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Trains the RoBERTa discriminator and finds the optimal probability threshold (τ).

    Args:
        train_df: DataFrame for training.
        val_df: DataFrame for validation and threshold selection.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def _encode(df):
        return tokenizer(df["message"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)

    ds_train = SMSDataset(_encode(train_df), train_df["label"].tolist())
    ds_val = SMSDataset(_encode(val_df), val_df["label"].tolist())

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=2e-5, # <--- CHANGE: Lowered from 5e-5 to 2e-5
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.2,
        logging_dir=LOG_DIR,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
    )

    trainer = PlainTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting discriminator training...")
    trainer.train()
    print("Training finished.")

    # --- OOD Score Thresholding (as per paper) ---
    # The "OOD score" is the model's predicted probability for the 'spam' class.
    # We find the threshold τ that maximizes F1 on the validation set.
    print("\nFinding optimal OOD score threshold (τ) on the validation set...")
    val_predictions = trainer.predict(ds_val)
    val_logits = val_predictions.predictions
    val_probs_spam = torch.softmax(torch.tensor(val_logits), dim=1)[:, 1].numpy()

    best_f1, best_tau = -1.0, 0.5
    y_true = val_df["label"].values

    # Iterate over potential thresholds to find the best one
    for tau in np.linspace(0.05, 0.95, 181): # Finer grid for better precision
        y_pred = (val_probs_spam >= tau).astype(int)
        # Calculate F1 for the 'spam' class (pos_label=1)
        f1 = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)[2]
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    print(f"\nSelected Threshold τ = {best_tau:.4f} (achieved max F1-score of {best_f1:.4f} on validation set)")

    # --- Persist model, tokenizer, and the threshold ---
    model_save_path = os.path.join(MODEL_DIR, "best_model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
        json.dump({"tau": best_tau}, f)

    print(f"Model and tokenizer saved to: {model_save_path}")
    print(f"Threshold saved to: {os.path.join(MODEL_DIR, 'threshold.json')}")


def main_train():
    """CLI entry point for the training and threshold selection stage."""
    print("\n--- Running Stage 2: Train Discriminator & Find Threshold ---")
    train_path = os.path.join(PROCESSED_DATA_DIR, AUGMENTED_TRAIN_FILE)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing augmented training file: {train_path}. Please run the 'augment' stage first.")

    df_train_all = pd.read_csv(train_path).dropna(subset=['message', 'label'])
    df_train_all['label'] = df_train_all['label'].astype(int)

    # Stratified split to ensure validation set has a similar label distribution
    train_df, val_df = train_test_split(
        df_train_all, test_size=0.1, random_state=SEED, stratify=df_train_all["label"]
    )

    print(f"Loaded augmented data. Total examples: {len(df_train_all)}")
    print(f"  Training set size: {len(train_df)}")
    print(f"  Validation set size: {len(val_df)}")

    fit_and_pick_threshold(train_df, val_df)


# =================================================================_
# STAGE 3: Evaluate on Test Set
# =================================================================_

def main_eval():
    """CLI entry point for the final evaluation stage."""
    print("\n--- Running Stage 3: Final Evaluation ---")
    test_path = os.path.join(PROCESSED_DATA_DIR, TEST_INPUT_FILE)
    model_path = os.path.join(MODEL_DIR, "best_model")
    threshold_path = os.path.join(MODEL_DIR, "threshold.json")

    # --- Validate paths ---
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}. Please run the 'train' stage first.")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}. Please run the 'train' stage first.")

    # --- Load test data ---
    df_test = pd.read_csv(test_path).dropna(subset=['message', 'label'])
    df_test['label'] = df_test['label'].astype(int)
    print(f"Loaded test data from {test_path}. Test set size: {len(df_test)}")

    # --- Load model and threshold τ ---
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    with open(threshold_path) as f:
        tau = json.load(f)["tau"]
    print(f"Loaded model and tokenizer from {model_path}")
    print(f"Using classification threshold τ = {tau:.4f}")

    # --- Make predictions ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    test_messages = df_test["message"].tolist()
    y_true = df_test["label"].values
    y_pred = []

    with torch.no_grad():
        # Process in batches to handle large test sets
        for i in range(0, len(test_messages), EVAL_BATCH_SIZE):
            batch = test_messages[i:i+EVAL_BATCH_SIZE]
            encodings = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH)
            encodings = {k: v.to(device) for k, v in encodings.items()}

            logits = model(**encodings).logits
            # Calculate spam probability (our "OOD score")
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            # Apply threshold τ to get final predictions
            preds = (probs >= tau).astype(int)
            y_pred.extend(preds)

    print("\n--- Test Set Evaluation Results ---")
    print(classification_report(y_true, y_pred, digits=4))


# =================================================================_
# Main Runner
# =================================================================_

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GLASS-FOOD pipeline runner for SMS Spam Detection.")
    parser.add_argument("stage", choices=["augment", "train", "eval", "all"], help="Which stage to run. 'all' runs them in sequence.")
    args = parser.parse_args()

    if args.stage == "augment":
        main_augment()
    elif args.stage == "train":
        main_train()
    elif args.stage == "eval":
        main_eval()
    elif args.stage == "all":
        main_augment()
        main_train()
        main_eval()