import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util  # For cosine similarity
import pandas as pd
import os
import re

# --- Configuration ---
# Model names (as per paper: BERT for initial, RoBERTa for generation/discrimination)
# For initial embedding to find farthest word, a general BERT should be fine.
# For generation (MLM), RoBERTa is specified.
BERT_MODEL_NAME = "bert-base-uncased"  # For farthest word embedding
ROBERTA_MLM_MODEL_NAME = "roberta-base"  # For masked word prediction

PROCESSED_DATA_DIR = "../../data/processed/"  # Relative path
TRAIN_SMS_FILE = "train_sms.csv"
AUGMENTED_TRAIN_SMS_FILE = "train_sms_augmented.csv"

# For PyTorch device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Load Models and Tokenizers (globally for efficiency if script is run multiple times) ---
try:
    # For Farthest Word (using BERT embeddings)
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    bert_model.eval()  # Set to evaluation mode

    # For Masked Word Prediction/Replacement (using RoBERTa MLM)
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MLM_MODEL_NAME)
    roberta_mlm_model = AutoModel.from_pretrained(ROBERTA_MLM_MODEL_NAME).to(DEVICE)  # Use AutoModelForMaskedLM for MLM
    # Correction: Need AutoModelForMaskedLM for MLM task
    from transformers import AutoModelForMaskedLM

    roberta_mlm_model = AutoModelForMaskedLM.from_pretrained(ROBERTA_MLM_MODEL_NAME).to(DEVICE)
    roberta_mlm_model.eval()  # Set to evaluation mode

    print("Successfully loaded models and tokenizers.")
except Exception as e:
    print(f"Error loading models: {e}. Make sure you have an internet connection for the first download.")
    bert_tokenizer, bert_model, roberta_tokenizer, roberta_mlm_model = None, None, None, None


# --- Helper Functions ---
# --- Helper Functions ---
def get_bert_embeddings(text, tokenizer, model):
    """Gets BERT embeddings for tokens and the [CLS] token (sentence embedding)."""
    if not text or not isinstance(text, str) or str(
            text).strip() == "":  # Added check for empty string after potential str conversion
        # print(f"Warning: Empty or invalid text received in get_bert_embeddings: '{text}'")
        return None, None, None  # <<< MODIFIED: Return three Nones

    # Ensure text is a string before tokenizing, even if it passed the initial check
    # This handles cases like text being a float NaN that was converted to "nan" string
    text_str = str(text)
    if not text_str.strip():  # Double check for empty string after str()
        return None, None, None

    try:
        inputs = tokenizer(text_str, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state.squeeze(0)
        sentence_embedding = token_embeddings[0]
        return token_embeddings, sentence_embedding, inputs['input_ids'].squeeze(0)
    except Exception as e:
        # print(f"Error during BERT embedding for text '{text_str[:50]}...': {e}")
        return None, None, None  # Return three Nones on any exception during processing


def get_word_token_map(text, token_ids, tokenizer):
    """Maps tokens back to words and their original indices. This is tricky!
       A simpler approach for now: operate on tokens directly if mapping is too complex.
       For this implementation, we'll find the farthest *token* and replace it.
       The paper is a bit ambiguous on "word" vs "token".
       If a "word" splits into multiple tokens, we'd pick the first token of that word.
    """
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    # Filter out special tokens like [CLS], [SEP], [PAD] for "farthest word" logic
    special_tokens = set(tokenizer.all_special_tokens)
    word_tokens_indices = [
        (i, token) for i, token in enumerate(tokens)
        if token not in special_tokens and not token.startswith("##")
        # Exclude subword continuations for "start of word"
    ]
    # A more advanced mapping would be needed for true word-level analysis.
    # For now, we consider each non-special, non-continuation token as a candidate.
    return word_tokens_indices


def find_farthest_token_index(token_embeddings, sentence_embedding, input_token_ids, tokenizer):
    """
    Finds the index of the token whose embedding is farthest from the sentence embedding.
    Considers only non-special, non-padding tokens.
    """
    if token_embeddings is None or sentence_embedding is None:
        return -1

    min_similarity = float('inf')
    farthest_idx_in_sequence = -1

    tokens = tokenizer.convert_ids_to_tokens(input_token_ids.tolist())
    special_tokens_ids = set(tokenizer.all_special_ids)

    for i in range(token_embeddings.size(0)):
        token_id = input_token_ids[i].item()
        if token_id in special_tokens_ids or tokens[i] == tokenizer.pad_token:  # Skip special and padding tokens
            continue

        # Consider only the first sub-word token if a word is split (simplification)
        # Or, for simplicity in this first pass, treat each token as a candidate.
        # The paper states "farthest word". If a word like "unbelievable" becomes "un", "##be", "##lie", "##vable",
        # we should ideally consider the embedding of "unbelievable" or the first token "un".

        current_token_embedding = token_embeddings[i]
        similarity = util.pytorch_cos_sim(current_token_embedding, sentence_embedding).item()

        if similarity < min_similarity:
            min_similarity = similarity
            farthest_idx_in_sequence = i  # This is the index in the tokenized sequence

    return farthest_idx_in_sequence


def replace_farthest_token(original_text, bert_tokenizer_inst, bert_model_inst,
                           roberta_tokenizer_inst, roberta_mlm_model_inst):
    """
    Implements the augmentation:
    1. Find farthest token using BERT embeddings.
    2. Mask it.
    3. Get RoBERTa MLM predictions for the mask.
    4. Select the best replacement based on cosine similarity to the original masked token's embedding.
    """
    if not all([bert_tokenizer_inst, bert_model_inst, roberta_tokenizer_inst, roberta_mlm_model_inst]):
        print("Models not loaded. Cannot augment.")
        return original_text  # Return original if models aren't ready

    # 1. Farthest Word Computation (using BERT)
    token_embeddings, sentence_embedding, input_token_ids_bert = get_bert_embeddings(
        original_text, bert_tokenizer_inst, bert_model_inst
    )
    if token_embeddings is None:
        return original_text  # Could not get embeddings

    farthest_token_idx = find_farthest_token_index(
        token_embeddings, sentence_embedding, input_token_ids_bert, bert_tokenizer_inst
    )

    if farthest_token_idx == -1 or farthest_token_idx >= len(input_token_ids_bert):  # Check bounds
        # print(f"Could not find a suitable farthest token for: {original_text}")
        return original_text  # No suitable token found or error

    original_farthest_token_id = input_token_ids_bert[farthest_token_idx].item()
    original_farthest_token_embedding = token_embeddings[farthest_token_idx]  # Embedding from BERT

    # 2. Mask the token in the original sequence (for RoBERTa MLM)
    # We need to re-tokenize with RoBERTa as tokenization might differ slightly.
    roberta_inputs = roberta_tokenizer_inst(original_text, return_tensors="pt", truncation=True, max_length=512).to(
        DEVICE)
    roberta_input_ids = roberta_inputs["input_ids"].squeeze(0)

    # Find the corresponding token in RoBERTa's tokenization. This is tricky.
    # A simpler but less accurate way: mask the token at the same *relative* position if tokenizers are similar.
    # A better way: identify the original word, then find its tokens in RoBERTa's output.
    # For now, let's try to find the token that BERT identified within RoBERTa's tokenization.
    # This assumes the "farthest word" logic applies to a single token.

    # Convert BERT token index to RoBERTa token index.
    # This is a simplification. A robust solution would map words.
    # Let's find the RoBERTa token(s) corresponding to the BERT farthest token.
    # This is non-trivial. For now, we'll assume the same index if tokenization is similar,
    # OR we find the first occurrence of that token ID.
    # This part needs refinement for robustness.
    # A simpler approach for a first pass: find the token that BERT identified by its string value
    # in RoBERTa's tokenization of the original text.

    original_farthest_token_str = bert_tokenizer_inst.convert_ids_to_tokens([original_farthest_token_id])[0]

    # Try to find this token string in RoBERTa's tokenization of the original text
    roberta_tokens_str = roberta_tokenizer_inst.convert_ids_to_tokens(roberta_input_ids.tolist())

    mask_idx_roberta = -1
    # Iterate through RoBERTa tokens to find a match for the BERT farthest token string
    # This is still a heuristic. True word alignment is complex.
    # We are looking for the BERT token that was deemed "farthest"
    # We need to find where that same conceptual "word" or "token" exists in RoBERTa's tokenization.

    # Simplification: let's use the index found by BERT directly in RoBERTa,
    # assuming tokenizations are somewhat aligned for the position of the mask.
    # This is a MAJOR simplification and likely needs improvement.
    # A more robust approach:
    #   1. Get the word string that `original_farthest_token_id` belongs to.
    #   2. Find that word in the original text.
    #   3. Tokenize the original text with RoBERTa.
    #   4. Find the RoBERTa token(s) corresponding to that word. Mask the first one.

    if farthest_token_idx < len(roberta_input_ids):  # Crude check
        mask_idx_roberta = farthest_token_idx  # Use BERT's index directly, very approximate
    else:  # Fallback if index is out of bounds
        # Try to find the first non-special token to mask in RoBERTa if alignment fails
        for i, token_id_rob in enumerate(roberta_input_ids.tolist()):
            if token_id_rob not in roberta_tokenizer_inst.all_special_ids:
                mask_idx_roberta = i
                break
        if mask_idx_roberta == -1:  # Still no maskable token
            return original_text

    if roberta_input_ids[mask_idx_roberta].item() in roberta_tokenizer_inst.all_special_ids:
        # print(f"Attempted to mask a special token at RoBERTa index {mask_idx_roberta}. Skipping augmentation for: {original_text}")
        # Try to find the *next* non-special token if the aligned one is special
        found_non_special = False
        for i in range(mask_idx_roberta + 1, len(roberta_input_ids)):
            if roberta_input_ids[i].item() not in roberta_tokenizer_inst.all_special_ids:
                mask_idx_roberta = i
                found_non_special = True
                break
        if not found_non_special: return original_text

    masked_input_ids_roberta = roberta_input_ids.clone()
    # Ensure we are not masking a special token (again, more robust logic needed here)
    if masked_input_ids_roberta[mask_idx_roberta].item() in roberta_tokenizer_inst.all_special_ids:
        # print(f"Warning: Attempting to mask a special RoBERTa token for '{original_text}'. Skipping.")
        return original_text

    original_roberta_token_at_mask_id = masked_input_ids_roberta[
        mask_idx_roberta].item()  # Store the token ID we are masking
    masked_input_ids_roberta[mask_idx_roberta] = roberta_tokenizer_inst.mask_token_id

    # 3. Get RoBERTa MLM predictions
    with torch.no_grad():
        outputs_mlm = roberta_mlm_model(input_ids=masked_input_ids_roberta.unsqueeze(0))  # Add batch dim
    predictions = outputs_mlm.logits.squeeze(0)  # Remove batch dim
    predicted_token_logits_at_mask = predictions[mask_idx_roberta]  # Logits for the masked position

    # 4. Select best replacement based on cosine similarity to ORIGINAL BERT farthest token's embedding
    # The paper says: "cosine similarity between wi and wj. Cosine similarity is
    # a metric used in machine learning to measure how similar two items are
    # irrespective of their size, and in this context, it quantifies the similarity
    # in the contextual usage of words."
    # This implies we need embeddings for candidate replacement tokens from RoBERTa's vocab.

    top_k = 10  # Consider top k predictions from MLM
    top_k_token_ids = torch.topk(predicted_token_logits_at_mask, top_k).indices.tolist()

    best_replacement_id = -1
    max_replacement_similarity = -float('inf')

    # Get embeddings for candidate RoBERTa tokens
    # We need the RoBERTa model's embedding layer
    roberta_word_embeddings = roberta_mlm_model.roberta.get_input_embeddings()  # Or equivalent for roberta-base

    for candidate_token_id in top_k_token_ids:
        if candidate_token_id == original_roberta_token_at_mask_id:  # Don't replace with itself
            continue
        if candidate_token_id in roberta_tokenizer_inst.all_special_ids:  # Don't replace with special token
            continue

        candidate_token_embedding = roberta_word_embeddings(torch.tensor([candidate_token_id]).to(DEVICE)).squeeze(0)
        similarity_to_original_farthest = util.pytorch_cos_sim(
            candidate_token_embedding,
            original_farthest_token_embedding  # This is from BERT
        ).item()

        if similarity_to_original_farthest > max_replacement_similarity:
            max_replacement_similarity = similarity_to_original_farthest
            best_replacement_id = candidate_token_id

    if best_replacement_id != -1:
        final_input_ids = roberta_input_ids.clone()  # Start with original RoBERTa tokenization
        final_input_ids[mask_idx_roberta] = best_replacement_id
        augmented_text = roberta_tokenizer_inst.decode(final_input_ids, skip_special_tokens=True)
        return augmented_text.strip()
    else:
        # print(f"Could not find a suitable replacement for: {original_text}")
        return original_text  # Fallback to original if no good replacement


def augment_dataset(df_train):
    """Augments the training dataframe."""
    if not all([bert_tokenizer, bert_model, roberta_tokenizer, roberta_mlm_model]):
        print("Models not loaded. Cannot augment dataset.")
        return df_train  # Return original if models aren't ready

    augmented_messages = []
    print(f"Starting augmentation for {len(df_train)} messages...")
    count = 0
    for index, row in df_train.iterrows():
        original_message = row['message']
        augmented_message = replace_farthest_token(
            original_message,
            bert_tokenizer, bert_model,
            roberta_tokenizer, roberta_mlm_model
        )
        augmented_messages.append(augmented_message)
        count += 1
        if count % 100 == 0:
            print(f"Augmented {count}/{len(df_train)} messages...")
            # print(f"  Original: {original_message[:60]}")
            # print(f"  Augmented: {augmented_message[:60]}")

    df_augmented_part = pd.DataFrame({
        'label': df_train['label'],  # Labels are inherited
        'message': augmented_messages
    })

    # Combine original training data with the augmented part
    # The paper doubles the dataset: 4458 original + 4458 generated = 8916
    df_combined_train = pd.concat([df_train, df_augmented_part], ignore_index=True)
    print(
        f"Finished augmentation. Original train size: {len(df_train)}, Augmented train size: {len(df_combined_train)}")
    return df_combined_train


if __name__ == "__main__":
    print("--- Phase 1: Text Augmentation (Generator) ---")

    train_file_path = os.path.join(PROCESSED_DATA_DIR, TRAIN_SMS_FILE)
    if not os.path.exists(train_file_path):
        print(f"Error: Processed training file not found at {train_file_path}")
        print("Please run the data_processing script first.")
    else:
        df_train_original = pd.read_csv(train_file_path)
        print(f"Loaded original training data: {df_train_original.shape}")
        # print(df_train_original.head())

        # Ensure models are loaded
        if not all([bert_tokenizer, bert_model, roberta_tokenizer, roberta_mlm_model]):
            print("Exiting due to model loading failure.")
        else:
            df_train_final_augmented = augment_dataset(df_train_original.copy())  # Use .copy()

            # Save the fully augmented training set
            augmented_train_path = os.path.join(PROCESSED_DATA_DIR, AUGMENTED_TRAIN_SMS_FILE)
            df_train_final_augmented.to_csv(augmented_train_path, index=False)
            print(f"Augmented training data saved to {augmented_train_path}")
            print(df_train_final_augmented.head())
            print(f"Final augmented training data shape: {df_train_final_augmented.shape}")
            print(
                f"Label distribution in augmented data:\n{df_train_final_augmented['label'].value_counts(normalize=True)}")