import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


EXPECTED_COLUMNS = [
    "location",
    "price",
    "bedrooms",
    "bathrooms",
    "area_sqft",
    "has_pool",
    "has_garage",
    "has_parking",
    "has_garden",
    "has_lake_view",
]


def _coerce_bool(value):
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        value = value.strip().lower()
        return 1 if value in {"1", "true", "yes", "y"} else 0
    return int(bool(value))


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def prepare_data(path: str | Path) -> pd.DataFrame:
    """Load and clean raw data."""
    df = load_dataset(path)
    df = df.copy()
    df["location"] = df["location"].fillna("Unknown")
    for col in ["bedrooms", "bathrooms", "area_sqft", "price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["has_pool", "has_garage", "has_parking", "has_garden", "has_lake_view"]:
        df[col] = df[col].apply(_coerce_bool)
    df = df.dropna(subset=["price", "bedrooms", "bathrooms", "area_sqft"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific features for better prediction.
    
    Adds:
    - price_per_sqft: normalized pricing metric
    - rooms_per_sqft: room density
    - amenity_count: total amenities
    - bedroom_bathroom_ratio: room balance
    - area_category: size buckets
    - location_encoded: location as ordinal or one-hot
    """
    df = df.copy()
    
    # 1. Price per square foot (market indicator)
    df["price_per_sqft"] = df["price"] / df["area_sqft"]
    
    # 2. Total rooms
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    
    # 3. Room density (rooms per 1000 sqft)
    df["room_density"] = (df["total_rooms"] / df["area_sqft"]) * 1000
    
    # 4. Bedroom-to-bathroom ratio (layout quality)
    df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 0.1)
    
    # 5. Total amenities (composite)
    amenity_cols = ["has_pool", "has_garage", "has_parking", "has_garden", "has_lake_view"]
    df["amenity_count"] = df[amenity_cols].sum(axis=1)
    
    # 6. Premium amenities (pool + lake view are high-value)
    df["premium_amenities"] = df["has_pool"] + df["has_lake_view"]
    
    # 7. Parking & garage combined
    df["parking_score"] = df["has_garage"] + df["has_parking"]
    
    # 8. Area category (bucketing size into tiers)
    df["area_category"] = pd.cut(
        df["area_sqft"],
        bins=[0, 1500, 2000, 2500, np.inf],
        labels=["small", "medium", "large", "xlarge"],
        ordered=True
    )
    
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Build feature matrix with engineered features and polynomial terms."""
    df_model = df.copy()
    
    # Apply feature engineering
    df_model = engineer_features(df_model)
    
    # Separate target
    y = df_model["price"]
    
    # Drop original price and non-feature columns
    X = df_model.drop(columns=[
        "price",
        "price_per_sqft",  # Don't use price-derived features in model
        "area_category"     # Will be one-hot encoded separately
    ])
    
    # One-hot encode location
    X = pd.get_dummies(X, columns=["location"], drop_first=False, dtype=int)
    
    # One-hot encode area_category
    area_dummies = pd.get_dummies(
        df_model["area_category"],
        prefix="area",
        drop_first=True,
        dtype=int
    )
    X = pd.concat([X, area_dummies], axis=1)
    
    # Polynomial features for key numeric columns (degree 2)
    # Focus on: area_sqft, bedrooms, bathrooms, total_rooms, room_density
    numeric_cols = ["area_sqft", "bedrooms", "bathrooms", "total_rooms", "room_density"]
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X[numeric_cols])
    poly_feature_names = poly.get_feature_names_out(numeric_cols)
    
    # Create polynomial feature dataframe
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
    
    # Combine polynomial features with other features
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
    X_final = pd.concat([X[non_numeric_cols], poly_df], axis=1)
    
    return X_final, y, poly, poly_feature_names, numeric_cols
