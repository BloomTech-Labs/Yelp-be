"""functions working with fasttext and LIME for the flask API."""

# imports
import boto3
import fasttext
import nltk
import re
import lime.lime_text
import numpy as np

import os
import tarfile

from dotenv import load_dotenv

def get_fasttext_model():
    """retrieve the stars classifier and it's dependencies, produce a classifier object."""

    load_dotenv()

    # download nltk
    nltk.download("punkt")

    # initialize s3 connection
    S3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("SPENCERS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SPENCERS_SECRET_ACCESS_KEY")
    )

    # download model files
    S3.download_file('yelp-dataset-pt-9', 'spencer/models/blazingtext/nltk/6m/stars/BlazingText-6m-stars/output/model.tar.gz', './ftmodels/stars.tar.gz')
    #S3.download_file('yelp-dataset-pt-9', 'spencer/models/blazingtext/nltk/6m/pos_neg/BlazingText-6m-pos-neg/output/model.tar.gz', './ftmodels/pos_neg.tar.gz')

    # unzip model files
    tar = tarfile.open('./ftmodels/stars.tar.gz', "r:gz")
    tar.extractall(path='./ftmodels')
    tar.close()

    # load model
    classifier = fasttext.load_model("./ftmodels/model.bin")

    # print a finish message.
    print('fasttext loaded!')

    return classifier


# This function regularizes a piece of text 
def strip_formatting(string: str) -> str:
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string


def tokenize_string(string: str) -> str:
    return string.split()



def get_fasttext_explainer():
    """returns a fasttext explainer object."""
    explainer = lime.lime_text.LimeTextExplainer(
        split_expression=tokenize_string,
        bow=False,
        class_names=["No Stars", "1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
    )
    return explainer


def fasttext_prediction_in_sklearn_format(classifier, texts):
    res = []
    labels, probabilities = classifier.predict(texts, 10)
    for label, probs, text in zip(labels, probabilities, texts):
        order = np.argsort(np.array(label))
        res.append(probs[order])

    return np.array(res)


def interpret(model, explainer, review: str, star_rating: int) -> dict:
    """interpret a review with respect to the given star rating."""

    exp = explainer.explain_instance(
        strip_formatting(review),
        classifier_fn=lambda x: fasttext_prediction_in_sklearn_format(model, x),
        top_labels=5,
        num_features=20,
    )

    interpretation = exp.as_list(star_rating-1)

    return dict(interpretation)