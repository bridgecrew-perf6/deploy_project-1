import sys

from ml.helper import get_categorical_features

from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump
from helper import process_data, train_model, get_categorical_features
# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('data/clean/census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.

train, test = train_test_split(data, test_size=0.2)

cat_features = get_categorical_features()

# Proces the test data with the process_data function.

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label='salary', training=True\
)

# Train and save a model.
trained_model = train_model(X_train, y_train)

dump(trained_model, "data/model/model.joblib")
dump(encoder, "data/model/encoder.joblib")
dump(lb, "data/model/lb.joblib")