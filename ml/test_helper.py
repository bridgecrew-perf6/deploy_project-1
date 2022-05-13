import pandas as pd
import numpy as np
import pytest
from ml.helper import process_data, get_categorical_features
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
    Check 
    """
    encoder_test = load("data/model/encoder.joblib")