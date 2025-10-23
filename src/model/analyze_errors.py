import pandas as pd
from transformers import pipeline, AutoTokenizer
import torch

# 1. Load your best model and tokenizer
MODEL_PATH = "../../models/global_best_discriminator/best_model" # Or point to the best iter
TEST_FILE_PATH = "../../data/processed/test_sms_dedup.csv"
DEVICE = 0 if torch.cuda.is_available() else -1

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = pipeline("text-classification", model=MODEL_PATH, tokenizer=tokenizer, device=DEVICE)

# 2. Load your test data
df_test = pd.read_csv(TEST_FILE_PATH)
df_test = df_test.dropna(subset=['message'])
spam_test_messages = df_test[df_test['label'] == 1]['message'].tolist()

print(f"Found {len(spam_test_messages)} spam messages in the test set. Predicting...")

# 3. Get predictions for all spam messages in the test set
# We use a pipeline for simplicity and batching
results = model(spam_test_messages, batch_size=128)

# 4. Find the False Negatives (the ones the model missed)
false_negatives = []
for message, result in zip(spam_test_messages, results):
    # The pipeline returns {'label': 'LABEL_1', 'score': 0.99}
    # We want the cases where it predicted LABEL_0 for a known spam message
    if result['label'] == 'LABEL_0':
        false_negatives.append(message)

# 5. Analyze the results
print("\n" + "="*50)
print(f"Analysis Complete. The model missed {len(false_negatives)} out of {len(spam_test_messages)} spam messages.")
print("These are the False Negatives from the TEST set:")
print("="*50)

for i, msg in enumerate(false_negatives):
    print(f"{i+1}: {msg}")

print("\n" + "="*50)
print("Look for patterns: Are they about a specific topic (packages, banks)? Do they lack URLs? Are they short?")