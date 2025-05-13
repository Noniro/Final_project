import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split # To create a validation set
import torch
import numpy as np
import os

# --- Configuration ---
PROCESSED_DATA_DIR = "../../data/processed/"
AUGMENTED_TRAIN_FILE = "train_sms_augmented.csv"
TEST_FILE = "test_sms.csv" # Original test set for final evaluation
MODEL_OUTPUT_DIR = "../../models/discriminator/"
LOGGING_DIR = '../../logs/discriminator_logs/'

ROBERTA_MODEL_NAME = "roberta-base"
MAX_LENGTH = 512 # As per paper
RANDOM_SEED = 42 # For reproducibility in splits and training

# Adjust based on your GPU memory and desired training speed
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 64
NUM_TRAIN_EPOCHS = 5 # Start with a reasonable number, paper doesn't specify epochs for this fine-tuning
LEARNING_RATE = 5e-5 # Common starting LR for fine-tuning transformers, adjust if needed
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1 # Warmup for 10% of total training steps, simpler than fixed 10k steps for fine-tuning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Load Data ---
def load_data(file_path):
    """Loads a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data from {file_path}: {e}")
        return None

# --- 2. Tokenize and Prepare Datasets ---
class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # The encodings from tokenizer are already dictionaries of lists.
        # We need to select the item at idx for each key.
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- 3. Define Metrics ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    # pos_label=1 assumes 'spam' is encoded as 1
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1, zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

if __name__ == "__main__":
    # --- Ensure augmented data exists ---
    augmented_train_path = os.path.join(PROCESSED_DATA_DIR, AUGMENTED_TRAIN_FILE)
    df_train_val = load_data(augmented_train_path) # This will be split into train and validation

    test_path = os.path.join(PROCESSED_DATA_DIR, TEST_FILE)
    df_test = load_data(test_path)

    if df_train_val is None or df_test is None:
        print("Exiting due to data loading errors.")
        exit()

    # --- Prepare texts and labels ---
    # Ensure 'message' and 'label' columns exist and are of correct type
    try:
        train_val_texts = df_train_val['message'].astype(str).tolist()
        train_val_labels = df_train_val['label'].astype(int).tolist()

        test_texts = df_test['message'].astype(str).tolist()
        test_labels = df_test['label'].astype(int).tolist()
    except KeyError as e:
        print(f"Error: Missing column in loaded data: {e}. Ensure 'message' and 'label' columns exist.")
        exit()
    except ValueError as e:
        print(f"Error: Could not convert labels to int: {e}. Check label column content.")
        exit()


    # --- Split augmented training data into actual training and a validation set ---
    # The paper doesn't explicitly mention a validation set for this phase, but it's good practice.
    # We'll use 10% of the augmented data for validation.
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=0.1, # 10% for validation
        random_state=RANDOM_SEED,
        stratify=train_val_labels # Important for imbalanced data
    )
    print(f"Augmented data split: {len(train_texts)} train, {len(val_texts)} validation, {len(test_texts)} test.")

    # --- Initialize Tokenizer and Model ---
    print(f"Loading tokenizer: {ROBERTA_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)

    print(f"Loading model for sequence classification: {ROBERTA_MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME,
        num_labels=2, # ham / spam
        # The paper doesn't specify ignoring mismatches, but it can be useful if you adapt a checkpoint
        # ignore_mismatched_sizes=True
    ).to(DEVICE)

    # --- Tokenize Data ---
    print("Tokenizing datasets...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = SMSDataset(train_encodings, train_labels)
    val_dataset = SMSDataset(val_encodings, val_labels)
    test_dataset = SMSDataset(test_encodings, test_labels)

    # --- Training Arguments ---
    # The paper's LR schedule (10k warmup steps to 1e-4) seems more for a larger pre-training or
    # different task. For fine-tuning on this specific (though augmented) dataset,
    # a smaller number of epochs and a more standard fine-tuning LR might be better.
    # We'll use a learning rate and warmup ratio common for fine-tuning.
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE, # Paper mentions 1e-4 target after warmup. Let's use a common fine-tuning LR.
        warmup_ratio=WARMUP_RATIO,   # Warmup for 10% of training steps
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGGING_DIR,
        logging_steps=50,         # Log metrics every 50 steps
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save model at the end of each epoch
        load_best_model_at_end=True, # Load the best model based on 'metric_for_best_model'
        metric_for_best_model="f1",  # Use F1-score on validation set to select best model
        greater_is_better=True,
        report_to="none",            # Can be "tensorboard", "wandb", etc.
        seed=RANDOM_SEED,
        # fp16=torch.cuda.is_available(), # Enable mixed precision training if on GPU, can speed up and save memory
    )

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, # Use the validation set for evaluation during training
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)] # Stop if f1 doesn't improve
    )

    # --- Train ---
    print("Starting Discriminator Training...")
    train_result = trainer.train()
    print("Training completed.")
    print(f"Training summary: {train_result}")


    # --- Evaluate on the held-out Test Set using the best model ---
    print("\nEvaluating on the Test Set using the best loaded model...")
    test_results = trainer.evaluate(eval_dataset=test_dataset) # Pass the actual test_dataset here

    print("\nTest Set Evaluation Results (Best Model):")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # --- Save the best model & tokenizer ---
    # The Trainer already saved the best model if load_best_model_at_end=True
    # But we can explicitly save it again if needed, or just confirm its location.
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_discriminator")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"Best discriminator model and tokenizer saved to {best_model_path}")

    # --- Optional: Make predictions on test set to see some examples ---
    # print("\nMaking some predictions on the test set:")
    # predictions_output = trainer.predict(test_dataset)
    # predicted_labels = np.argmax(predictions_output.predictions, axis=1)
    # for i in range(min(10, len(test_texts))):
    #     print(f"Text: {test_texts[i][:100]}...")
    #     print(f"Actual: {'Spam' if test_labels[i]==1 else 'Ham'}, Predicted: {'Spam' if predicted_labels[i]==1 else 'Ham'}")
    #     print("-" * 20)