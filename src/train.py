import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import logging
import os

from data.data_preprocess import preprocess_for_bert

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Siêu tham số
batch_size = 64
epochs = 3
max_length = 32
learning_rate = 2e-5

# Đường dẫn dữ liệu và nơi lưu mô hình
file_path = '/Users/apple/Downloads/GitHub/TitleClassification/data/News Title.xls'
save_path = 'saved_model'
os.makedirs(save_path, exist_ok=True)


def evaluate(model, val_dataloader):
    model.eval()
    val_preds, val_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_masks, labels = [item.to(device) for item in batch]

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            val_preds.extend(predictions.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(val_labels, val_preds)
    avg_loss = total_loss / len(val_dataloader)
    return acc, avg_loss, val_preds, val_labels


def plot_metrics(train_loss_list, val_loss_list, val_acc_list):
    epochs_range = range(1, len(train_loss_list) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    plt.plot(epochs_range, val_loss_list, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_acc_list, label='Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


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
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []

    logger.info("Starting training...")
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            input_ids, attention_masks, labels = [item.to(device) for item in batch]

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        train_loss_list.append(avg_train_loss)

        val_acc, val_loss, val_preds, val_labels = evaluate(model, val_dataloader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            logger.info("Best model saved.")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("Final model & tokenizer saved to 'saved_model/'")

    # Đánh giá trực quan
    label_names = list(dict_labels.keys())

    logger.info("Plotting metrics...")
    plot_metrics(train_loss_list, val_loss_list, val_acc_list)

    cm = confusion_matrix(val_labels, val_preds)
    plt.figure()
    plot_confusion_matrix(cm, figsize=(8, 6), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(len(label_names)), label_names, fontsize=12)
    plt.yticks(range(len(label_names)), label_names, fontsize=12)
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=label_names))


if __name__ == "__main__":
    main()
