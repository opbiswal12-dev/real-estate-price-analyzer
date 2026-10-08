import argparse
import pickle
from pathlib import Path

from src.data_processor import prepare_data
from src.model import train_and_score_model
from src.scraper import scrape_real_estate_page


DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "price_model.pkl"


def train_pipeline(data_path: str = "data/sample_houses.csv") -> dict:
    df = prepare_data(data_path)
    metrics = train_and_score_model(df)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(metrics["model"], f)
    return metrics


def predict_price(args) -> float:
    if not MODEL_PATH.exists():
        train_pipeline()

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    feature_values = {
        "location": args.location,
        "bedrooms": args.bedrooms,
        "bathrooms": args.bathrooms,
        "area_sqft": args.area_sqft,
        "has_pool": int(args.has_pool),
        "has_garage": int(args.has_garage),
        "has_parking": int(args.has_parking),
        "has_garden": int(args.has_garden),
        "has_lake_view": int(args.has_lake_view),
    }

    prediction = model.predict_single(feature_values)
    print(f"Predicted house price: ${prediction:,.2f}")
    return prediction


def command_scrape(args):
    output_path = args.output or "data/scraped_houses.csv"
    df = scrape_real_estate_page(args.url, output_path)
    print(f"Saved {len(df)} rows to {output_path}")


def command_train(args):
    metrics = train_pipeline(args.data)
    print(f"R²: {metrics['r2']:.4f}")
    print(f"MAE: ${metrics['mae']:,.2f}")
    print(f"RMSE: ${metrics['rmse']:,.2f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Real Estate Price Analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser("scrape", help="Scrape real-estate listing data")
    scrape_parser.add_argument("--url", required=True, help="Target listing page URL")
    scrape_parser.add_argument("--output", default="data/scraped_houses.csv", help="Output CSV path")
    scrape_parser.set_defaults(func=command_scrape)

    train_parser = subparsers.add_parser("train", help="Train the regression model")
    train_parser.add_argument("--data", default="data/sample_houses.csv", help="Input CSV dataset")
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser("predict", help="Predict price for a single house")
    predict_parser.add_argument("--location", required=True, help="Property location")
    predict_parser.add_argument("--area_sqft", type=float, required=True)
    predict_parser.add_argument("--bedrooms", type=float, required=True)
    predict_parser.add_argument("--bathrooms", type=float, required=True)
    predict_parser.add_argument("--has_pool", type=int, default=0)
    predict_parser.add_argument("--has_garage", type=int, default=0)
    predict_parser.add_argument("--has_parking", type=int, default=0)
    predict_parser.add_argument("--has_garden", type=int, default=0)
    predict_parser.add_argument("--has_lake_view", type=int, default=0)
    predict_parser.set_defaults(func=predict_price)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
