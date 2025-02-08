from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from datasets import load_dataset

ds = load_dataset("yassiracharki/Amazon_Reviews_for_Sentiment_Analysis_fine_grained_5_classes")
# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("path/to/model/folder")
model.eval()  # Set to evaluation mode

# Sample input text
text = "This is a phishing email, please do not click any links."

# Prepare input
inputs = tokenizer(text, return_tensors="pt")

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Convert logits to probabilities
probabilities = torch.softmax(logits, dim=1)

# Get predicted class
predicted_class = torch.argmax(probabilities, dim=1).item()

# Output result
print(f"Predicted Class: {predicted_class}, Probabilities: {probabilities}")
