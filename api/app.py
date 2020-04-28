import json

from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd


from api.helper_functions import infer, squish


def create_app():
    # Elastic Beanstalk initalization
    application = Flask(__name__)
    CORS(application)

    @application.route('/')
    def root():
        return 'yelpsense api. hit /infer_sentiment/?review= for inference.'

    @application.route('/infer_sentiment/', methods=['GET'])
    def infer_sentiment():
        """return review sentiment prediction between zero and one."""
        review = request.args.get('review')

        return jsonify(dict(zip(['sentiment'],[squish(infer(review))])))

    return application

if __name__ == '__main__':
    application = create_app()