import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_processor import build_feature_matrix


@dataclass
class RegressionModel:
    model: LinearRegression
    feature_columns: list

    def predict_single(self, feature_dict: dict) -> float:
        row = {key: [value] for key, value in feature_dict.items()}
        df = pd.DataFrame(row)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        prediction = self.model.predict(df)
        return float(prediction[0])


def train_and_score_model(df: pd.DataFrame):
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    model_wrapper = RegressionModel(model=model, feature_columns=list(X.columns))

    return {
        "model": model_wrapper,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "feature_columns": list(X.columns),
    }
