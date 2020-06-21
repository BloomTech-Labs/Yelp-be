from marshmallow import Schema, fields, ValidationError


def validate_model(model):
    if model not in ('t5', 'bart'):
        raise ValidationError("Model not one of 't5' or 'bart'")


class SummarySchema(Schema):
    text = fields.Str(required=True)
    model = fields.Str(required=True, validate=validate_model)
