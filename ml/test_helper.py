import pandas as pd
import numpy as np
import pytest
from helper import process_data, get_categorical_features, inference
from joblib import load


@pytest.fixture
def data():
    """
    Get the data
    """
    df = pd.read_csv("data/clean/census.csv")
    return df


def test_process_data(data):
    """
    Check train and test have same number of rows for X and y
    """
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    X_test, y_test, _, _ = process_data(
        data,
        categorical_features=get_categorical_features(),
        label="salary",
        encoder=encoder,
        lb=lb,
        training=False
    )

    assert(len(X_test) == len(y_test))


def test_process_encoder(data):
    """
    Check encoder structure
    """
    encoder_test = load("data/model/encoder.joblib")
    lb_test = load("data/model/lb.joblib")

    _, _, encoder, lb = process_data(
        data,
        categorical_features=get_categorical_features(),
        label="salary",
        training=True
    )

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()


def test_inference_above():
    """
    Check inference performance
    """

    model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    array = np.array([[
        40,
        "Private",
        "Some-college",
        "Married-civ-spouse",
        "Exec-managerial",
        "Husband",
        "Black",
        "Male",
        80,
        "United-States"
    ]])

    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = process_data(
        df_temp,
        categorical_features=get_categorical_features(),
        encoder=encoder,
        lb=lb,
        training=False
    )

    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"


def test_inference_below():
    """
    Check inference performance
    """
    model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    array = np.array([[
                     19,
                     "Private",
                     "HS-grad",
                     "Never-married",
                     "Own-child",
                     "Husband",
                     "Black",
                     "Male",
                     30,
                     "United-States"
                     ]])
    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = process_data(
        df_temp,
        categorical_features=get_categorical_features(),
        encoder=encoder, lb=lb, training=False)

    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"
