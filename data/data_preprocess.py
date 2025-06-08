import torch
from transformers import BertTokenizer
from data_loader import data_load, clean_data
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bien tieu de thanh input_ids va attention_masks phu hop voi input cua BERT
def encode_titles_for_bert(titles, tokenizer, max_length=64):
    encoded_data = tokenizer.batch_encode_plus(
        titles.tolist(),
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']

    return input_ids, attention_masks

# Tao tap du lieu TensorDataset cho BERT
def create_bert_dataset(input_ids, attention_masks, labels):
    """Create TensorDataset for BERT inputs"""
    return torch.utils.data.TensorDataset(
        input_ids,
        attention_masks,
        torch.tensor(labels, dtype=torch.long)
    )

# Tai và tiền xử lý dữ liệu cho BERT, tra ve train_dataset, val_dataset, dict_labels
def preprocess_for_bert(file_path, test_size=0.2, max_length=64):

    # Load & clean data
    logger.info("Loading data...")
    titles, labels, dict_labels = data_load(file_path)
    logger.info("Cleaning data...")
    titles = clean_data(titles)

    # Split data
    logger.info("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(titles, labels, test_size=test_size, random_state=42)

    # Tokenizer
    logger.info("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Encode titles
    logger.info("Encoding training data...")
    train_input_ids, train_attention_masks = encode_titles_for_bert(X_train, tokenizer, max_length)
    logger.info("Encoding validation data...")
    val_input_ids, val_attention_masks = encode_titles_for_bert(X_val, tokenizer, max_length)

    # Create datasets
    train_dataset = create_bert_dataset(train_input_ids, train_attention_masks, y_train)
    val_dataset = create_bert_dataset(val_input_ids, val_attention_masks, y_val)

    logger.info("Preprocessing complete.")
    return train_dataset, val_dataset, dict_labels