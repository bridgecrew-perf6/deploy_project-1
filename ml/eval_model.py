import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
from ml.helper import get_categorical_features, process_data, compute_model_metrics
import logging

def evaluate_model():
    df = pd.read_csv("data/clean/census.csv")
    _, test = train_test_split(df, test_size=0.2, random_state=42)

    trained_model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    slice_values = []

    categories = get_categorical_features()

    for cat in categories:
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=categories,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False
            )

            y_preds = trained_model.predict(X_test)

            prc, rcl, fb = compute_model_metrics(y_test, y_preds)

            line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)
            logging.info(line)
            slice_values.append(line)

    with open('data/model/slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')