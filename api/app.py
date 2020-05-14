import json

from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd


from api.distilbert_functions import get_distilBERT_objects, infer, squish, starify
from api.fasttext_functions import get_fasttext_model, get_fasttext_explainer, interpret


def create_app():
    # app initalization
    application = Flask(__name__)
    CORS(application)

    # get DistilBERT model objects
    (dBERT_tokenizer, dBERT_model) = get_distilBERT_objects()

    # get model interpretability objects
    ft_classifier = get_fasttext_model()
    ft_explainer = get_fasttext_explainer()


    @application.route('/')
    def root():
        return 'yelpsense api. hit /infer_sentiment/?review= for inference.'
    

    @application.route('/infer_sentiment/', methods=['GET'])
    def infer_sentiment():
        """return review sentiment prediction between zero and one, an estimation of star rating, and class interpretation."""

        review = request.args.get('review')
        inference = infer(dBERT_model,dBERT_tokenizer,review)
        sentiment = squish(inference)
        star_rating = starify(inference)
        interpretation = interpret(ft_classifier,ft_explainer,review,star_rating)

        return jsonify(dict( zip(['sentiment', 'star_rating', 'class_interpretation'],
                                 [sentiment, star_rating, interpretation]) ))

    return application