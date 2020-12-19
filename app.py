# app.py
from flask import Flask
from flask_restful import Api, Resource, reqparse
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib
import numpy as np
import json
import re

APP = Flask(__name__)
API = Api(APP)

true_fake_model = joblib.load('true-fake.mdl')
stopwords = np.genfromtxt('ukrainian-stopwords.txt',dtype='str')
corpus = pd.read_csv('corpus.csv', index_col=False)['text']

class Predict(Resource):

    def clean_train_data(x):
        text = str(x)
        text = text.lower()
        text = re.sub('\[.*?\]', '', text) # remove square brackets
        text = re.sub(r'[^\w\s]','',text) # remove punctuation
        text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
        text = re.sub(r'http\S+', '', text)
        text = re.sub('\n', '', text)
        return text

    def remove_ukr_stopwords(text):
        token_text = text.split(' ')
        remove_stop = [word for word in token_text if word not in stopwords]
        join_text = ' '.join(remove_stop)
        return join_text

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('title')
        parser.add_argument('text')

        args = parser.parse_args()  # creates dict
        combine_text = args['title'] + ' ' + args['text']
        combine_text = Predict.clean_train_data(combine_text)
        combine_text = Predict.remove_ukr_stopwords(combine_text)

        X_test = []
        X_test.append(combine_text)

        vec_train = CountVectorizer().fit(corpus)
        X_vec_train = vec_train.transform(corpus)
        X_vec_test = vec_train.transform(X_test)

        out = true_fake_model.predict_proba(X_vec_test)
        out = {'Probability of fake': out[0, 0], 'Probability of true': out[0, 1]}

        return out, 200


API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True, port='1080')
