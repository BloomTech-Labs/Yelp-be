from flask import Flask, request
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np

app = Flask(__name__)

sentiment = pipeline("sentiment-analysis")
bart_summarization = pipeline("summarization")
t5_summarization = pipeline(
    "summarization", model="t5", tokenizer="t5", framework="tf")


tokenizer = AutoTokenizer.from_pretrained(
    'distilbert-base-uncased', use_fast=True)
model = TFAutoModelForSequenceClassification.from_pretrained(
    "models/distilbert/regression")


def predict(text):
    return


def scale(pred):
    return


def squish(scaled):
    return


def pred_scale_and_squish(text):
    pred = model(tokenizer.encode(text, return_tensors='tf', max_length=512))[
        0].numpy()[0][0]
    scaled = (pred - 1) * .25
    squished = np.clip(scaled, 0, 1)

    return pred, scaled, squished


def old_summarize(ts, text, **generate_kwargs):
    # Add prefix to text
    prefix = ts.model.config.prefix if ts.model.config.prefix is not None else ""
    documents = (prefix + text,)

    # tokenize
    inputs = ts.tokenizer.encode_plus(
        *documents,
        return_tensors='tf',
        max_length=ts.tokenizer.max_len
    )

    summaries = ts.model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"], **generate_kwargs,
    )
    results = []
    for summary in summaries:
        record = {}
        record["summary_text"] = ts.tokenizer.decode(
            summary, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )

        results.append(record)
    return results


@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    text = request.form['text']
    return sentiment(text)[0]


@app.route('/summarization', methods=['POST'])
def get_summarization():
    text = request.form['text']
    return summarization(text)[0]


@app.route('/')
def hello_world():
    return 'Hello, World!'
