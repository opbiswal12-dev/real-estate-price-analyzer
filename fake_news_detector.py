import argparse
import joblib
from pathlib import Path

from src.fake_news_model import (
    FakeNewsDetector,
    load_model_bundle,
    predict_news_article,
    train_and_evaluate_model,
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "fake_news_model.joblib"


def train_command(args):
    bundle = train_and_evaluate_model(args.dataset)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Training complete. Model saved to {MODEL_PATH}")
    print(f"Accuracy: {bundle['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {bundle['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {bundle['metrics']['roc_auc']:.4f}")


def predict_command(args):
    if not MODEL_PATH.exists():
        print("Model not found. Training a fresh model...")
        bundle = train_and_evaluate_model(args.dataset)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
    else:
        bundle = load_model_bundle(MODEL_PATH)

    result = predict_news_article(args.text, bundle)
    print(f"Prediction: {'REAL NEWS' if result['label'] == 0 else 'FAKE NEWS'}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print(f"Probability: {result['probability']:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Advanced Fake News Detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the fake news detection model")
    train_parser.add_argument("--dataset", default="data/fake_news_dataset.csv", help="Training CSV dataset")
    train_parser.set_defaults(func=train_command)

    predict_parser = subparsers.add_parser("predict", help="Predict if a news article is fake or real")
    predict_parser.add_argument("--text", required=True, help="News article text to classify")
    predict_parser.add_argument("--dataset", default="data/fake_news_dataset.csv", help="Dataset path if a model needs to be retrained")
    predict_parser.set_defaults(func=predict_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
