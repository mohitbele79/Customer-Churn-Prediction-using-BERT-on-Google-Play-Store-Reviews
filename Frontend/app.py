from flask import Flask, render_template, request, redirect, url_for
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os

app = Flask(__name__)

# Create upload directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load BERT model and tokenizer
model_path = r"../my_bert_model/bert_churn_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)


# Redirect root to single prediction

@app.route('/')
def home():
    return redirect(url_for('single_prediction'))


# Single Review Prediction Route
@app.route('/single_prediction', methods=['GET', 'POST'])
def single_prediction():
    prediction = None
    review = None

    if request.method == 'POST':
        review = request.form['text']
        if review.strip():
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            prediction = ["Positive", "Negative"][pred]

    return render_template('single_prediction.html', prediction=prediction, review=review)

# Batch Review Prediction Route
@app.route('/batch_prediction', methods=['GET', 'POST'])
def batch_prediction():
    predictions = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            if 'review' not in df.columns:
                return "CSV file must contain a 'review' column."

            results = []
            for text in df['review']:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                results.append(["Positive", "Negative"][pred])

            df["Prediction"] = results
            predictions = df.to_html(classes="table", index=False)
        else:
            return "Please upload a valid .csv file with a 'review' column."

    return render_template('batch_prediction.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
