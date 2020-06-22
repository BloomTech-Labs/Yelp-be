from flask import Flask, request
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np

app = Flask(__name__)


""" Load all the models and such """


default_sentiment = pipeline("sentiment-analysis")
bart_summarization = pipeline("summarization")
t5_summarization = pipeline(
    "summarization", model="t5-small", tokenizer="t5-small", framework="tf")


distilbert_tokenizer = AutoTokenizer.from_pretrained(
    'distilbert-base-uncased', use_fast=True)
distilbert_regression_model = TFAutoModelForSequenceClassification.from_pretrained(
    "models/distilbert/regression")


""" End of loading models and such """
""" Define prediction and helper function """


class Summarization:
    def __init__(self, model):
        self.model = model

    def __call__(self, text, **generate_kwargs):
        # Add prefix to text
        prefix = self.model.model.config.prefix if self.model.model.config.prefix is not None else ""
        documents = (prefix + text,)

        # tokenize
        inputs = self.model.tokenizer.encode_plus(
            *documents,
            return_tensors=self.model.framework,
            max_length=self.model.tokenizer.max_len
        )

        summaries = self.model.model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"], **generate_kwargs,
        )
        results = []
        for summary in summaries:
            record = {}
            record["summary_text"] = self.model.tokenizer.decode(
                summary, skip_special_tokens=True, clean_up_tokenization_spaces=True,
            )

            results.append(record)
        return results


def distilbert_regression(text):
    pred = distilbert_regression_model(distilbert_tokenizer.encode(text, return_tensors='tf', max_length=512))[
        0].numpy()[0][0]
    scaled = (pred - 1) * .25
    squished = np.clip(scaled, 0, 1)

    if squished > 0.5:
        label = "POSITIVE"
    else:
        label = "NEGATIVE"

    res = {
        'label': label,
        'score': float(squished),
        'star': float(pred)
    }

    return [res]


SUPPORTED_MODELS = {
    'summarization': {
        'bart': Summarization(bart_summarization),
        't5': Summarization(t5_summarization)
    },
    'sentiment': {
        'distilbert-regression': distilbert_regression,
        'default': default_sentiment
    }
}


def validate_required_inputs(request):
    required = ['text', 'model_name']
    missing = missing = [
        field for field in required if field not in request.form.keys()]
    if missing:
        err = {field: f"the {field} field is required" for field in missing}
        return err, 422


def validate_model_name(request, model_type):
    model_name = request.form['model_name']
    if model_name not in SUPPORTED_MODELS[model_type].keys():
        err = {
            'model_name': f"{model_name} is not one of the supported models. {list(SUPPORTED_MODELS[model_type].keys())}"
        }
        return err, 422


def validate(request, model_type):
    valid_required = validate_required_inputs(request)
    if valid_required:
        return valid_required

    valid_model = validate_model_name(request, model_type)
    if valid_model:
        return valid_model


def get_result(request, model_type):
    valid = validate(request, model_type)
    if valid:
        return valid

    text = request.form['text']
    model_name = request.form['model_name']

    model = SUPPORTED_MODELS[model_type][model_name]

    return model(text)[0]


""" end of define prediction and helper function """
""" Routes """


@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    return get_result(request, 'sentiment')


@app.route('/summarization', methods=['POST'])
def get_summarization():
    return get_result(request, 'summarization')


@app.route('/')
def hello_world():
    return 'Hello, World!'
