from flask import Flask, request, jsonify
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from flask_cors import CORS

import pandas

from dotenv import load_dotenv

from lightfm_functions import get_lightfm_model, get_lightfm_dataset, make_lightfm_user_set, lightfm_inference, select_from_db

import os


app = Flask(__name__)
CORS(app)
load_dotenv()


""" Load all the models and such """

default_sentiment = pipeline("sentiment-analysis")
bart_summarization = pipeline("summarization")
t5_summarization = pipeline(
    "summarization", model="t5-small", tokenizer="t5-small", framework="tf")


distilbert_tokenizer = AutoTokenizer.from_pretrained(
    'distilbert-base-uncased', use_fast=True)
distilbert_regression_model = TFAutoModelForSequenceClassification.from_pretrained(
    "models/distilbert/regression")

# make a directory to download lightfm models to
if not os.path.exists('lightfm'):
    os.makedirs('lightfm')

# get lightFM model and dataset objects
lightFM_model = get_lightfm_model()
lightFM_dataset = get_lightfm_dataset()


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
        field for field in required if field not in request.keys()]
    if missing:
        err = {field: f"the {field} field is required" for field in missing}
        return err, 422


def validate_model_name(request, model_type):
    model_name = request['model_name']
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
    req_data = request.get_json()
    valid = validate(req_data, model_type)
    if valid:
        return valid

    text = req_data['text']
    model_name = req_data['model_name']

    model = SUPPORTED_MODELS[model_type][model_name]

    if model_type == 'summarization':
        min_length = model.model.model.config.min_length
        max_length = model.model.model.config.max_length
        if 'min_length' in req_data:
            min_length = req_data['min_length']
            if min_length < 10:
                return {'min_length': 'Min length is too small'}
        if 'max_length' in req_data:
            if req_data['max_length'] > model.model.tokenizer.max_len:
                return {'max_length': 'Max length is too big'}
            max_length = req_data['max_length']

        return model(text, min_length=min_length, max_length=max_length)[0]

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

@app.route('/business_info/', methods=['GET'])
def business_info():
    """search the business db for businesses matching query parameters."""

    city = request.args.get('city')
    name = request.args.get('name')
    address = request.args.get('address')
    categories = request.args.get('categories')

    if city is None: city = ''
    if name is None: name = ''
    if address is None: address = ''
    if categories is None: categories = ''


    response = select_from_db(city=city, business_name=name, address=address, category=categories)
    labeled = [dict(zip(['business_id', 'name', 'address', 'city', 'aggregate_rating', 'categories'],element)) for element in response]
    return jsonify(labeled)

@app.route('/infer_recommendations/', methods=['GET'])
def infer_recommendations():
    """given a list of business ids, infer recommendations."""

    train_business_ids = request.args.getlist('business_ids')
    train_stars = [int(x) for x in request.args.getlist('stars')]

    user_set, user_shape = make_lightfm_user_set(dataset=lightFM_dataset,businesses=train_business_ids, stars=train_stars)

    predictions = lightfm_inference(lightFM_model, user_set, user_shape)

    output = [dict(zip(['business_id', 'name', 'address', 'city', 'aggregate_rating', 'categories', 'recommender_score'],element)) for element in predictions.itertuples(index=False)]

    return jsonify(output)
