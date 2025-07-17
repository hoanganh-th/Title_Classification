import torch
from transformers import BertForSequenceClassification, BertTokenizer
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to saved model
MODEL_PATH = '/Users/apple/Downloads/GitHub/TitleClassification/src/saved_model'
MAX_LENGTH = 32  # Should match the value used during training

# Define the labels dictionary (must match what was used during training)
dict_labels = {'Entertainment': 0, 'Business': 1, 'Technology': 2, 'Medical': 3}
# Invert the dictionary to map from index to label name
idx_to_label = {v: k for k, v in dict_labels.items()}


def clean_shortforms(text):
    short_forms_dict = {
        "ain't": "is not", "aren't": "are not", "can't": "cannot",
        "'cause": "because", "could've": "could have", "couldn't": "could not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
        "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
        "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
        "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
        "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
        "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
        "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
        "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
        "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
        "she'll've": "she will have", "she's": "she is", "should've": "should have",
        "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
        "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
        "that's": "that is", "there'd": "there would", "there'd've": "there would have",
        "there's": "there is", "here's": "here is", "they'd": "they would",
        "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
        "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
        "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
        "we'll've": "we will have", "we're": "we are", "we've": "we have",
        "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
        "what're": "what are", "what's": "what is", "what've": "what have",
        "when's": "when is", "when've": "when have", "where'd": "where did",
        "where's": "where is", "where've": "where have", "who'll": "who will",
        "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
        "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
        "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
        "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
        "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
        "you're": "you are", "you've": "you have"
    }

    clean_text = text
    for shortform in short_forms_dict.keys():
        if re.search(shortform, text):
            clean_text = re.sub(shortform, short_forms_dict[shortform], text)
    return clean_text


def clean_symbol(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text):
    """Preprocess text using the same steps as during training"""
    # Convert to string, just in case
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Handle contractions and short forms
    text = clean_shortforms(text)
    # Remove special characters
    text = clean_symbol(text)
    return text


def load_model():
    """Load the pre-trained model and tokenizer"""
    try:
        logger.info("Loading model from {}".format(MODEL_PATH))
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        # Load model
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None


def predict_title(title, model, tokenizer):
    """Predict the category of a news title"""
    # Preprocess the title
    processed_title = preprocess_text(title)
    
    # Tokenize
    encoded_title = tokenizer.encode_plus(
        processed_title,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded_title['input_ids'].to(device)
    attention_mask = encoded_title['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    predicted_category = idx_to_label[prediction]
    
    # Get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence = probabilities[0][prediction].item()
    
    return predicted_category, confidence


def main():
    """Main function to run the prediction program"""
    print("Loading model, please wait...")
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Make sure you have trained the model first.")
        return
    
    print("\n=== News Title Classifier ===")
    print("Type 'exit' to quit the program")
    
    while True:
        print("\nEnter a news title:")
        title = input("> ")
        
        if title.lower() == 'exit':
            print("Exiting program...")
            break
        
        if not title.strip():
            print("Please enter a valid title.")
            continue
        
        category, confidence = predict_title(title, model, tokenizer)
        print(f"\nPredicted Category: {category}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")


if __name__ == "__main__":
    main()
