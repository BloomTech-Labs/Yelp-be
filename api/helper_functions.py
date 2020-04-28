"""helper functions for the flask API."""
import numpy as np

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    'distilbert-base-uncased', use_fast=True)

model = TFAutoModelForSequenceClassification.from_pretrained(
    "spentaur/yelp")

def infer(review: str) -> float:
    """returns raw model sentiment prediction for a review."""
    return model(tokenizer.encode(review, return_tensors='tf'))[0].numpy()[0][0]

def squish(x: float) -> float:
    """converts model inferences to value between zero and one."""
    return np.clip(0.25 * x + 0.5, 0, 1)