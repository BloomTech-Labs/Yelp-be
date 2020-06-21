import os
import tarfile
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline


class Predictor:
    def __init__(self, config):
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            print("Has AWS Access")
            # client will use your credentials if available
            s3 = boto3.client("s3")
        else:
            print("Doesn't Have AWS Access")
            s3 = boto3.client("s3", config=Config(
                signature_version=UNSIGNED))  # anonymous client

        model_tar_path = "/tmp/model.tar.gz"
        model_path = "/tmp/model"

        s3.download_file(config["bucket"], config["key"], model_tar_path)

        with tarfile.open(model_tar_path, "r:gz") as tar:
            tar.extractall(path=model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased', use_fast=True)
        model = TFAutoModelForSequenceClassification.from_pretrained(
            "/tmp/model")

        self.model = model
        self.tokenizer = tokenizer
        self.t5_summarizer = pipeline(
            "summarization", model="t5-large", tokenizer="t5-large", framework="tf")
        self.bart_summarizer = pipeline("summarization")

    def _summarize(self, ts, text, **generate_kwargs):
        # Add prefix to text
        prefix = ts.model.config.prefix if ts.model.config.prefix is not None else ""
        documents = (prefix + text,)

        # tokenize
        inputs = ts.tokenizer.encode_plus(
            *documents,
            return_tensors=ts.framework,
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

    def _scale(self, pred):
        return (pred - 1) * .25

    def _squish(self, scaled):
        return np.clip(scaled, 0, 1)

    def _predict_star(self, text):
        return self.model(self.tokenizer.encode(text, return_tensors='tf', max_length=512))[0].numpy()[0][0]

    def t5_summary(self, text):
        return self._summarize(self.t5_summarizer, text)[0]['summary_text']

    def bart_summary(self, text):
        return self._summarize(self.bart_summarizer, text)[0]['summary_text']

    def sentiment(self, text):
        star_pred = self._predict_star(text)
        star_scaled = self._scale(star_pred)
        star_squished = self._squish(star_scaled)

        res = {
            'star_pred': float(star_pred),
            'star_scaled': float(star_scaled),
            'star_squished': float(star_squished)
        }

        return res
