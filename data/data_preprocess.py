import torch
from transformers import BertTokenizer
from data_loader import data_load, clean_data
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bien tieu de thanh input_ids va attention_masks phu hop voi input cua BERT
def encode_titles_for_bert(titles, tokenizer, max_length=128):
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