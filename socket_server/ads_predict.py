import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize

# Load tokenizer và model từ Hugging Face
tokenizer = AutoTokenizer.from_pretrained("Khoa/kompa-check-ads-0725", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("Khoa/kompa-check-ads-0725")

device = 0 if torch.cuda.is_available() else -1
text_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)


def preprocess_text(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    return ' '.join(tokens)


def predict_ads(text: str) -> bool:
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    processed_text = preprocess_text(text)
    result = text_classifier(processed_text, truncation=True, max_length=100)[0]

    label_id = int(result['label'].split('_')[-1]) if "label" in result['label'].lower() else 0

    is_ads = True if label_id == 1 else False
    return is_ads
