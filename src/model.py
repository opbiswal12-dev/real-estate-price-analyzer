import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from src.data_processor import build_feature_matrix, engineer_features


@dataclass
class RegressionModel:
    """Wrapper for regression model with preprocessing and prediction."""
    model: Ridge
    feature_columns: list
    scaler: StandardScaler
    poly: PolynomialFeatures
    poly_feature_names: list
    numeric_cols: list
    location_columns: list
    area_columns: list

    def predict_single(self, feature_dict: dict) -> float:
        """Predict price for a single property."""
        # Build feature row
        row_data = {
            "location": feature_dict["location"],
            "bedrooms": feature_dict["bedrooms"],
            "bathrooms": feature_dict["bathrooms"],
            "area_sqft": feature_dict["area_sqft"],
            "has_pool": int(feature_dict.get("has_pool", 0)),
            "has_garage": int(feature_dict.get("has_garage", 0)),
            "has_parking": int(feature_dict.get("has_parking", 0)),
            "has_garden": int(feature_dict.get("has_garden", 0)),
            "has_lake_view": int(feature_dict.get("has_lake_view", 0)),
        }
        
        # Create a small dataframe and engineer features
        df_temp = pd.DataFrame([row_data])
        df_temp = engineer_features(df_temp)
        
        # Build feature vector
        X_temp = pd.DataFrame()
        
        # Add engineered features
        for col in ["total_rooms", "room_density", "bed_bath_ratio", 
                   "amenity_count", "premium_amenities", "parking_score"]:
            X_temp[col] = df_temp[col]
        
        # Add boolean features
        for col in ["has_pool", "has_garage", "has_parking", "has_garden", "has_lake_view"]:
            X_temp[col] = df_temp[col]
        
        # Add location one-hot encoding
        for loc_col in self.location_columns:
            X_temp[loc_col] = 1 if loc_col == f"location_{feature_dict['location']}" else 0
        
        # Add area category (default to medium for unknown)
        for area_col in self.area_columns:
            X_temp[area_col] = 0
        X_temp["area_medium"] = 1  # Default to medium if not specified
        
        # Extract numeric columns for polynomial features
        numeric_data = X_temp[self.numeric_cols].values
        
        # Create polynomial features
        poly_features = self.poly.transform(numeric_data)
        poly_df = pd.DataFrame(poly_features, columns=self.poly_feature_names)
        
        # Get non-numeric columns
        non_numeric_cols = [col for col in X_temp.columns if col not in self.numeric_cols]
        X_final = pd.concat([X_temp[non_numeric_cols], poly_df], axis=1)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in X_final.columns:
                X_final[col] = 0
        
        X_final = X_final[self.feature_columns]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_final)
        prediction = self.model.predict(X_scaled)
        
        return float(prediction[0])


def train_and_score_model(df: pd.DataFrame):
    """Train Ridge regression with engineered features."""
    X, y, poly, poly_feature_names, numeric_cols = build_feature_matrix(df)
    
    # Identify location and area columns for later use
    location_columns = [col for col in X.columns if col.startswith("location_")]
    area_columns = [col for col in X.columns if col.startswith("area_")]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge regression (handles multicollinearity from polynomial features)
    model = Ridge(alpha=100.0)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Wrap model
    model_wrapper = RegressionModel(
        model=model,
        feature_columns=list(X.columns),
        scaler=scaler,
        poly=poly,
        poly_feature_names=poly_feature_names,
        numeric_cols=numeric_cols,
        location_columns=location_columns,
        area_columns=area_columns,
    )
    
    return {
        "model": model_wrapper,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "feature_columns": list(X.columns),
    }
