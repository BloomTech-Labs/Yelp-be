import os

from flask import Flask
from flask import request, abort
from dotenv import load_dotenv

from api.FormSchemas.Sentiment import SentimentSchema
from api.FormSchemas.Summary import SummarySchema
from api.Predictor import Predictor

load_dotenv()
sentiment_schema = SentimentSchema()
summary_schema = SummarySchema()

app = Flask(__name__)

config = {
    'bucket': 'yelpsense',
    'key': 'models/sentiment/distilbert/regression/model.tar.gz'
}
predictor = Predictor(config)


@app.route('/sentiment', methods=['POST'])
def sentiment():
    errors = sentiment_schema.validate(request.form)
    if errors:
        return errors, 422

    text = request.form['text']
    return predictor.sentiment(text)


@app.route('/summary', methods=['POST'])
def summary():
    errors = summary_schema.validate(request.form)

    if errors:
        return errors, 422

    text = request.form['text']
    if request.form['model'] == 't5':
        return predictor.t5_summary(text)
    elif request.form['model'] == 'bart':
        return predictor.bart_summary(text)
