import os

from flask import Flask
from flask import request, abort
from dotenv import load_dotenv

from FormSchemas.Sentiment import SentimentSchema
from FormSchemas.Summary import SummarySchema
from Predictor import Predictor

load_dotenv()
sentiment_schema = SentimentSchema()
summary_schema = SummarySchema()

# EB looks for an 'application' callable by default.
application = Flask(__name__)

config = {
    'bucket': 'yelpsense',
    'key': 'models/sentiment/distilbert/regression/model.tar.gz'
}
predictor = Predictor(config)


@application.route('/sentiment', methods=['POST'])
def sentiment():
    errors = sentiment_schema.validate(request.form)
    if errors:
        return errors, 422

    text = request.form['text']
    return predictor.sentiment(text)


@application.route('/summary', methods=['POST'])
def summary():
    errors = summary_schema.validate(request.form)

    if errors:
        return errors, 422

    text = request.form['text']
    if request.form['model'] == 't5':
        return predictor.t5_summary(text)
    elif request.form['model'] == 'bart':
        return predictor.bart_summary(text)


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    # application.debug = True
    application.run(host="0.0.0.0")
