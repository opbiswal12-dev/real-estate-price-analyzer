from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents="unicode",
            stop_words="english",
        )
        self.model = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                )),
                ("lr", LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                )),
            ],
            voting="soft",
        )

    def _compute_text_features(self, texts):
        series = pd.Series(texts).fillna("")
        features = pd.DataFrame({
            "char_count": series.str.len(),
            "word_count": series.str.split().str.len(),
            "avg_word_length": series.apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0.0),
            "exclamation_count": series.str.count("!"),
            "question_count": series.str.count("?"),
            "uppercase_ratio": series.apply(lambda x: sum(1 for ch in x if ch.isupper()) / max(len(x), 1)),
            "digit_count": series.str.count(r"\d"),
        })
        return features.fillna(0.0).to_numpy(dtype=float)

    def fit(self, texts, labels):
        cleaned_texts = [self._normalize_text(t) for t in texts]
        self.vectorizer.fit(cleaned_texts)
        tfidf_train = self.vectorizer.transform(cleaned_texts)
        text_features = self._compute_text_features(cleaned_texts)
        X_train = hstack([tfidf_train, csr_matrix(text_features)])
        self.model.fit(X_train, labels)
        return self

    def predict(self, texts):
        cleaned_texts = [self._normalize_text(t) for t in texts]
        tfidf = self.vectorizer.transform(cleaned_texts)
        test_features = self._compute_text_features(cleaned_texts)
        X_test = hstack([tfidf, csr_matrix(test_features)])
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1]
        return predictions, probabilities

    @staticmethod
    def _normalize_text(text):
        text = str(text).lower()
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text


def load_dataset(path: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df["text"], df["label"].astype(int)


def train_and_evaluate_model(dataset_path: str = "data/fake_news_dataset.csv"):
    texts, labels = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    detector = FakeNewsDetector()
    detector.fit(X_train, y_train)

    predictions, probabilities = detector.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "classification_report": classification_report(y_test, predictions, target_names=["REAL", "FAKE"]),
    }

    bundle = {
        "detector": detector,
        "metrics": metrics,
    }
    return bundle


def predict_news_article(article_text: str, bundle: dict) -> dict:
    detector = bundle["detector"]
    prediction, probabilities = detector.predict([article_text])
    label = int(prediction[0])
    probability = float(probabilities[0])
    confidence = probability if label == 1 else 1.0 - probability
    return {
        "label": label,
        "probability": probability,
        "confidence": confidence,
    }


def save_model_bundle(bundle: dict, path: str):
    import joblib
    joblib.dump(bundle, path)


def load_model_bundle(path: str):
    import joblib
    return joblib.load(path)
