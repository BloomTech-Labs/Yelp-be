"""functions working with Distilbert for the flask API."""
# imports
import numpy as np

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def get_distilBERT_objects():
    """retrieve distilBERT tokenizer and model objects."""

    tokenizer = AutoTokenizer.from_pretrained(
        'distilbert-base-uncased', use_fast=True)

    model = TFAutoModelForSequenceClassification.from_pretrained(
        "spentaur/yelp")

    # print a finishing message
    print('distilbert loaded!')

    return (tokenizer, model)


def infer(model, tokenizer, review: str) -> float:
    """returns raw model sentiment prediction for a review."""
    return float(model(tokenizer.encode(review, return_tensors='tf'))[0].numpy()[0][0])

def starify(x: float) -> int:
    """change output of infer to a star rating guess."""
    return int(round(x+3))

def squish(x: float) -> float:
    """converts model inferences to value between zero and one."""
    return float(np.clip(0.25 * x + 0.5, 0, 1))