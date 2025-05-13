import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os

# --- Configuration ---
RAW_DATA_DIR = "../../data/raw/" # Relative to this script's location
#NUS_CORPUS_SUBDIR = "nus_sms_corpus/" # If NUS data is in a subfolder
PROCESSED_DATA_DIR = "../../data/processed/"

SMS_SPAM_COLLECTION_FILE = "SMSSpamCollection"
NUS_SQL_DUMP_FILE = "smsCorpus_en_2015.03.09_all.sql" # Name of your SQL dump file
# You might have multiple SQL files from NUS, this example assumes one main dump for simplicity

TRAIN_FILE_PROCESSED = "train_sms.csv"
TEST_FILE_PROCESSED = "test_sms.csv"
NUS_HAM_PROCESSED_FILE = "nus_ham_messages.csv" # Output for NUS ham messages

RANDOM_SEED = 42
TEST_SET_SIZE = 0.20

def load_sms_data(file_path):
    """Loads the SMS Spam Collection dataset."""
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
        print(f"Successfully loaded SMS Spam Collection data from {file_path}")
        print(f"Raw SMS Spam Collection data shape: {df.shape}")
        # print("Raw SMS Spam Collection data head:\n", df.head()) # Keep output concise
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading SMS Spam Collection data: {e}")
        return None

def parse_nus_sql_dump(file_path):
    """
    Parses the NUS SMS Corpus SQL dump to extract message content (assumed to be the 6th field).
    Handles multi-row INSERTs and avoids unsafe unicode decoding.
    """
    messages = []
    insert_prefix = "INSERT INTO `new_sms_download` VALUES"
    buffer = ""

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(insert_prefix):
                    buffer = line.strip()
                    while not buffer.endswith(";"):
                        buffer += f.readline().strip()

                    # Extract all tuples in the VALUES part
                    values_part = buffer[len(insert_prefix):].strip().rstrip(';')
                    value_tuples = re.findall(r"\((.*?)\)", values_part)

                    for tuple_str in value_tuples:
                        # Safe CSV-style split on comma while respecting quotes
                        fields = re.findall(r"(?:'[^']*'|[^,]+)", tuple_str)
                        if len(fields) >= 6:
                            msg = fields[5].strip()
                            if msg.startswith("'") and msg.endswith("'"):
                                msg = msg[1:-1]  # Strip surrounding quotes
                            msg = msg.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
                            messages.append(msg)

        print(f"Extracted {len(messages)} messages from NUS SQL dump.")
        return pd.DataFrame(messages, columns=['message'])

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame(columns=['message'])
    except Exception as e:
        print(f"Error while parsing SQL dump: {e}")
        return pd.DataFrame(columns=['message'])




def preprocess_text(text):
    """Basic text preprocessing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataframe_sms_spam(df):
    """Applies preprocessing to the SMS Spam Collection dataframe."""
    if df is None:
        return None
    df['message_cleaned'] = df['message'].apply(preprocess_text)
    df['label_numeric'] = df['label'].map({'ham': 0, 'spam': 1})
    # print("SMS Spam Collection data after preprocessing and label encoding:")
    # print(df[['label_numeric', 'message_cleaned']].head())
    # print(f"Value counts for labels:\n{df['label_numeric'].value_counts(normalize=True)}")
    return df

def preprocess_dataframe_nus(df_nus):
    """Applies preprocessing to the NUS ham messages dataframe."""
    if df_nus is None or df_nus.empty:
        print("NUS dataframe is None or empty. Skipping preprocessing.")
        return pd.DataFrame(columns=['label', 'message']) # Return empty df with target columns

    df_nus['message_cleaned'] = df_nus['message'].apply(preprocess_text)
    df_nus['label_numeric'] = 0 # All messages from NUS are 'ham' (label 0)
    # print("NUS data after preprocessing and label encoding:")
    # print(df_nus[['label_numeric', 'message_cleaned']].head())
    return df_nus[['label_numeric', 'message_cleaned']].rename(columns={'label_numeric': 'label', 'message_cleaned': 'message'})


def split_and_save_data(df, test_size, random_state, processed_dir, train_filename, test_filename):
    """Splits SMS Spam Collection data into training and testing sets and saves them."""
    if df is None or 'label_numeric' not in df.columns or 'message_cleaned' not in df.columns:
        print("Error: SMS Spam Collection dataframe is None or missing required columns for splitting.")
        return None, None
    try:
        X = df['message_cleaned']
        y = df['label_numeric']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        train_df = pd.DataFrame({'label': y_train, 'message': X_train})
        test_df = pd.DataFrame({'label': y_test, 'message': X_test})

        print(f"\nSMS Spam Training set shape: {train_df.shape}")
        print(f"SMS Spam Testing set shape: {test_df.shape}")
        # print(f"Training label distribution:\n{train_df['label'].value_counts(normalize=True)}")
        # print(f"Testing label distribution:\n{test_df['label'].value_counts(normalize=True)}")

        os.makedirs(processed_dir, exist_ok=True)
        train_path = os.path.join(processed_dir, train_filename)
        test_path = os.path.join(processed_dir, test_filename)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Processed SMS Spam training data saved to {train_path}")
        print(f"Processed SMS Spam testing data saved to {test_path}")
        return train_df, test_df
    except Exception as e:
        print(f"An error occurred during SMS Spam data splitting or saving: {e}")
        return None, None

def save_nus_data(df_nus_processed, processed_dir, nus_filename):
    """Saves the processed NUS ham messages."""
    if df_nus_processed is None or df_nus_processed.empty:
        print("No processed NUS data to save.")
        return
    try:
        os.makedirs(processed_dir, exist_ok=True)
        nus_path = os.path.join(processed_dir, nus_filename)
        df_nus_processed.to_csv(nus_path, index=False)
        print(f"Processed NUS ham data saved to {nus_path}")
        print(f"NUS Ham data shape: {df_nus_processed.shape}")
    except Exception as e:
        print(f"An error occurred while saving NUS data: {e}")


if __name__ == "__main__":
    print("--- Phase 0: Load and Preprocess Datasets ---")

    # 1. Process SMS Spam Collection
    print("\n--- Processing SMS Spam Collection ---")
    raw_sms_spam_file_path = os.path.join(RAW_DATA_DIR, SMS_SPAM_COLLECTION_FILE)
    sms_df_raw = load_sms_data(raw_sms_spam_file_path)

    if sms_df_raw is not None:
        sms_df_processed = preprocess_dataframe_sms_spam(sms_df_raw.copy())
        # if sms_df_processed is not None:
        #     train_data, test_data = split_and_save_data(
        #         sms_df_processed,
        #         test_size=TEST_SET_SIZE,
        #         random_state=RANDOM_SEED,
        #         processed_dir=PROCESSED_DATA_DIR,
        #         train_filename=TRAIN_FILE_PROCESSED,
        #         test_filename=TEST_FILE_PROCESSED
        #     )
            # if train_data is not None:
            #     print("\nSample of processed SMS Spam training data:")
            #     print(train_data.head())

    # 2. Process NUS SMS Corpus (from SQL dump)
    print("\n--- Processing NUS SMS Corpus (from SQL Dump) ---")
    nus_sql_file_path = os.path.join(RAW_DATA_DIR, NUS_SQL_DUMP_FILE)
    df_nus_raw = parse_nus_sql_dump(nus_sql_file_path)

    if not df_nus_raw.empty:
        df_nus_processed = preprocess_dataframe_nus(df_nus_raw.copy())

        if not df_nus_processed.empty and sms_df_processed is not None:
            # Combine both datasets (SMS Spam + NUS Ham)
            combined_df = pd.concat([
                sms_df_processed[['label_numeric', 'message_cleaned']],
                df_nus_processed.rename(columns={'label': 'label_numeric', 'message': 'message_cleaned'})
            ], ignore_index=True)

            # Split and save the combined dataset (same filenames as before)
            split_and_save_data(
                df=combined_df,
                test_size=TEST_SET_SIZE,
                random_state=RANDOM_SEED,
                processed_dir=PROCESSED_DATA_DIR,
                train_filename=TRAIN_FILE_PROCESSED,
                test_filename=TEST_FILE_PROCESSED
            )
        else:
            print("Warning: NUS or SMS Spam Collection data is empty. Skipping merging.")
    else:
        print("NUS data extraction resulted in an empty dataframe.")
