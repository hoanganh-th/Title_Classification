import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import logging
import os

from data.data_preprocess import preprocess_for_bert

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Siêu tham số
batch_size = 128
epochs = 1
max_length = 64
learning_rate = 2e-5

# Đường dẫn dữ liệu và nơi lưu mô hình
file_path = '/Users/apple/Downloads/GitHub/TitleClassification/data/News Title.xls'
save_path = 'saved_model'
os.makedirs(save_path, exist_ok=True)

def evaluate(model, val_dataloader):
    """Đánh giá mô hình trên tập validation"""
    model.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_masks, labels = [item.to(device) for item in batch]

            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            val_preds.extend(predictions.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(val_labels, val_preds)
    return acc

def main():
    logger.info("Preprocessing data...")
    train_dataset, val_dataset, dict_labels = preprocess_for_bert(file_path, max_length=max_length)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    logger.info("Loading model...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(dict_labels),
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    best_val_acc = 0.0

    logger.info("Starting training...")
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids, attention_masks, labels = [item.to(device) for item in batch]

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")

        val_acc = evaluate(model, val_dataloader)
        logger.info(f"Validation Accuracy: {val_acc:.4f}")

        # Lưu mô hình tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            logger.info("Best model saved.")

    # Lưu toàn bộ mô hình + tokenizer để tái sử dụng
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("Final model & tokenizer saved to 'saved_model/'")

if __name__ == "__main__":
    main()
