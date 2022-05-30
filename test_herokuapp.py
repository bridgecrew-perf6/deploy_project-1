"""
Heroku Api test script
"""
import requests
import pytest


@pytest.fixture
def test_herokuapp():
    data = {
        "age": 32,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    }

    r = requests.post('https://udacity-deploy.herokuapp.com/', json=data)

    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
