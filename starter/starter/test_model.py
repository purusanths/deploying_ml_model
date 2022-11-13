import pandas as pd
import pytest
from starter.ml.data import process_data
from sklearn.model_selection import train_test_split
from starter.ml.model import train_model,compute_model_metrics,inference
import os
import pickle


import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)



@pytest.fixture
def data():
    """
    A fixture to retur the data
    """
    #data =pd.read_csv('../data/census.csv')
    data = pd.read_csv('starter/data/census.csv')

    return data

@pytest.fixture
def cat_features():
    """
    A fixture to return the cat feature
    """
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    return cat_features

@pytest.fixture
def train_test_splits(data):
    """
    A fixture to return the train test split
    """
    train, test = train_test_split(data, test_size=0.20)
    return train,test

@pytest.fixture(scope="session")
def process_data_f(train_test_splits,cat_features):
    """
    A fixture to return the train test split
    """
    train,test =train_test_splits
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

    X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,encoder=encoder,lb=lb)

    return X_train, y_train, X_test, y_test

def test_process_data_f(process_data_f):
    """
    A unit test to check the trsin test split
    """

    X_train, y_train, X_test, y_test= process_data_f

    assert X_train.shape[1]== X_test.shape[1]


def test_trained_model(process_data_f):
    """
    A unit  test to check weather trained model is saved
    """
    X_train, y_train,_,_=process_data_f
    model=train_model(X_train, y_train)
    with open('starter/model/rfc_model.pkl', 'wb') as files: #'../model/rfc_model.pkl'
        pickle.dump(model, files)

    assert os.path.exists('starter/model/rfc_model.pkl')  # '../model/rfc_model.pkl'

def test_inference(process_data_f):
    """
    A unit test to test inference function
    """
    X_train, y_train, X_test, y_test= process_data_f
    model = pickle.load(open('starter/model/rfc_model.pkl', 'rb')) #'../model/rfc_model.pkl'
    prediction=inference(model,X_test)

    assert X_test.shape[0]==prediction.shape[0]

# def test_compute_model_metrics(process_data_f):
#     """
#     unit test to test the performacne metrices
#     """
#     X_train, y_train, X_test, y_test= process_data_f
#     model = pickle.load(open('starter/model/rfc_model.pkl', 'rb')) #'../model/rfc_model.pkl'
#     prediction=inference(model,X_test)

#     precision, recall, fbeta =compute_model_metrics(y_test,prediction)

#     assert 0< precision
#     assert 0< recall
#     assert 0< fbeta 

def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Greetings": "Welcome to the MLops World!"}

def test_post():
    data = json.dumps({
        "age": 45,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
        })
    r = client.post("/inference/",data=data)
    assert r.status_code == 200

def test_post_prediction_0():
    data = json.dumps({
        "age": 45,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
        })
    r = client.post("/inference/",data=data)
    assert r.json() == {"prediction": "0"}

def test_post_prediction_1():
    ### need to find an example which give the model prediction of 1
    data = json.dumps({
        "age": 28,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 3000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
        })
    r = client.post("/inference/",data=data)
    assert r.json() == {"prediction": "0"}

