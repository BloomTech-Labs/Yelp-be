from marshmallow import Schema, fields


class SentimentSchema(Schema):
    text = fields.Str(required=True)
