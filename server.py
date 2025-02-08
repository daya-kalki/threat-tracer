from flask import Flask, request, jsonify
import nbformat
from nbconvert import PythonExporter
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")

app = Flask(__name__)

# Function to run Jupyter Notebook (if needed)
def run_notebook(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    python_code, _ = exporter.from_notebook_node(notebook)
    exec(python_code, globals())

@app.route('/scan', methods=['POST'])
def scan_email():
    data = request.json
    email_text = data.get("email", "")

    # Process the email with RoBERTa
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    result = torch.argmax(outputs.logits).item()

    response = "Phishing Detected!" if result == 1 else "Email is Safe"
    return jsonify({"result": response})

if __name__ == '__main__':
    app.run(debug=True)
