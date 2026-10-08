import pandas as pd
from pathlib import Path


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
    df = load_dataset(path)
    df = df.copy()
    df["location"] = df["location"].fillna("Unknown")
    for col in ["bedrooms", "bathrooms", "area_sqft", "price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["has_pool", "has_garage", "has_parking", "has_garden", "has_lake_view"]:
        df[col] = df[col].apply(_coerce_bool)
    df = df.dropna(subset=["price", "bedrooms", "bathrooms", "area_sqft"])
    return df


def build_feature_matrix(df: pd.DataFrame):
    df_model = df.copy()
    df_model["location"] = df_model["location"].astype(str)
    X = pd.get_dummies(df_model.drop(columns=["price"]), columns=["location"], dtype=int)
    y = df_model["price"]
    return X, y
