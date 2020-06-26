"""functions working with the lightFM recommender model."""

# imports
import boto3

import lightfm
import pandas as pd
import numpy as np

import pickle

import psycopg2


from lightfm.data import Dataset

import os
import copy


# get model
# remember that you may need to use two dots for this to work


def get_lightfm_model():
    """unpickles a lightfm model and returns it."""

    # initialize s3 connection
    S3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY")
    )

    S3.download_file(
        'yelpsense', 'models/recommender/lightfm/lightfm_model.p', './lightfm/lightfm_model.p')

    model = pickle.load(open("./lightfm/lightfm_model.p", "rb"))
    return model


def get_lightfm_dataset():
    """unpickles a lightfm dataset with appropriate shape and returns it."""

    # initialize s3 connection
    S3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY")
    )

    S3.download_file('yelpsense', 'models/recommender/lightfm/lightfm_empty_dataset.p',
                     './lightfm/lightfm_empty_dataset.p')

    empty_dataset = pickle.load(
        open("./lightfm/lightfm_empty_dataset.p", "rb"))
    return empty_dataset


def make_lightfm_user_set(dataset, businesses, stars):
    """Makes lightFM interactions and a lightFM dataset of reviews given a list of businesses and star ratings."""

    """# make lists from user_dict
    businesses = user_dict.keys()
    stars = user_dict.values()"""

    # get a user's data
    userframe = pd.DataFrame({'business_id': businesses, 'stars': stars})
    userframe['recommend'] = userframe['stars'].map(
        {1: -1, 2: -1, 3: -1, 4: 1, 5: 1})
    userframe['user_id'] = 'new_user'

    # get an empty but shaped dataset and populate with interactions
    user_dataset = copy.deepcopy(dataset)
    (user_interactions, weights) = user_dataset.build_interactions([(x['user_id'],
                                                                     x['business_id'],
                                                                     x['recommend']) for index, x in userframe.iterrows()])

    return (user_interactions, user_dataset)


def make_business_dataframe():
    """Makes a dataframe of businesses to populate with predicted reviews."""

    conn = psycopg2.connect(host=os.getenv('AWS_RDS_HOST'),
                            database=os.getenv('AWS_RDS_DB'),
                            user=os.getenv('AWS_RDS_USER'),
                            password=os.getenv('AWS_RDS_PASS'),
                            port=os.getenv('AWS_RDS_PORT'))

    cur = conn.cursor()

    query = f"SELECT business_id, name, address, city, aggregate_rating, categories FROM business_data"

    businessframe = pd.read_sql(sql=query, con=conn)

    conn.close()

    return businessframe


def lightfm_inference(model, user_interactions, user_dataset):
    """does secondary training on a model with given data, and returns recommendations."""

    # copy the model object to prevent contamination
    pretrained = copy.deepcopy(model)

    pretrained.fit_partial(user_interactions, epochs=50)

    businessframe = make_business_dataframe()

    businessframe['lightFM_mapping'] = businessframe['business_id'].apply(
        lambda x: user_dataset._item_id_mapping[x])

    businessframe['recommender_values'] = model.predict(user_ids=[user_dataset._user_id_mapping['new_user']],
                                                        item_ids=list(businessframe['lightFM_mapping'].values), num_threads=1)

    top_ten = businessframe[['business_id', 'name', 'address', 'city', 'aggregate_rating',
                             'categories', 'recommender_values']].sort_values(by='recommender_values', ascending=False)

    return top_ten


def select_from_db(city='', business_name='', address='', category=''):
    """gets results for businesses from database based on params."""

    conn = psycopg2.connect(host=os.getenv('AWS_RDS_HOST'),
                            database=os.getenv('AWS_RDS_DB'),
                            user=os.getenv('AWS_RDS_USER'),
                            password=os.getenv('AWS_RDS_PASS'),
                            port=os.getenv('AWS_RDS_PORT'))

    cur = conn.cursor()

    query = f"SELECT business_id, name, address, city, aggregate_rating, categories FROM business_data WHERE city ILIKE '%{city}%' AND name ILIKE '%{business_name}%' AND address ILIKE '%{address}%' AND categories ILIKE '%{category}%'"

    cur.execute(query)
    output = cur.fetchall()

    conn.close()

    return output
