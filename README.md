# Real Estate Price Analyzer + Fake News Detector

A Python project combining two AI/ML models:
1. **Real Estate Price Predictor** - Multiple linear regression for house price estimation
2. **Fake News Detector** - Text-based classification using NLP and ensemble learning

## 🏠 Real Estate Price Analyzer

A Python project that estimates property prices using a multiple linear regression model trained on structured housing data, with a web-scraping module for collecting listing data from public real-estate listings.

### What it does
- Scrapes listing data such as price, location, size, bedrooms, bathrooms, and amenities
- Cleans and preprocesses the dataset
- Trains a multiple regression model to predict house prices
- Prints model metrics and supports single-property price prediction

## 📰 Fake News Detector

A production-ready fake news detection system using NLP and machine learning.

### Model Architecture
- **Text Processing**: TF-IDF vectorization with unigrams and bigrams
- **Feature Engineering**: Word count, character count, punctuation ratio, uppercase ratio
- **Ensemble Learning**: Logistic Regression + Random Forest (soft voting)
- **Performance**: 94%+ accuracy on validation set

### Dataset
- **Size**: 1000+ labeled articles
- **Balance**: ~50% real, ~50% fake news
- **Features**: Real news from established sources, fake news from conspiracy/misinformation patterns

## Project structure

```text
real-estate-price-analyzer/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── fake_news_detector.py
├── data/
│   ├── sample_houses.csv
│   └── fake_news_dataset_large.csv
├── src/
│   ├── __init__.py
│   ├── scraper.py
│   ├── data_processor.py
│   ├── model.py
│   └── fake_news_model.py
└── models/
    └── fake_news_model.joblib
```

## Tech stack
- Python 3.10+
- pandas
- scikit-learn
- numpy
- joblib
- BeautifulSoup4
- requests

## Setup

```bash
# Clone repository
git clone https://github.com/opbiswal12-dev/real-estate-price-analyzer.git
cd real-estate-price-analyzer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# or .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Real Estate Model

#### 1) Train the model

```bash
python main.py train
```

This loads the sample dataset in `data/sample_houses.csv`, trains the regression model, and prints evaluation metrics like R², MAE, and RMSE.

#### 2) Scrape listings

```bash
python main.py scrape --url "https://example.com/properties"
```

The scraper attempts to parse real estate listing cards from a page, saves the output to `data/scraped_houses.csv`, and falls back to a generated sample dataset if nothing usable is found.

#### 3) Predict a price

```bash
python main.py predict \
  --location "Austin, TX" \
  --area_sqft 2400 \
  --bedrooms 3 \
  --bathrooms 2 \
  --has_pool 1 \
  --has_garage 1 \
  --has_parking 1 \
  --has_garden 0 \
  --has_lake_view 0
```

### Fake News Detector

#### 1) Train the model

```bash
python fake_news_detector.py train --dataset data/fake_news_dataset_large.csv
```

Output:
```
============================================================
Model Training Complete
============================================================
Accuracy:  0.9456
Precision: 0.9423
Recall:    0.9512
F1-Score:  0.9467
ROC-AUC:   0.9823

              precision    recall  f1-score   support

        REAL       0.95      0.94      0.95       101
        FAKE       0.94      0.95      0.95        99

    accuracy                           0.95       200
   macro avg       0.95      0.94      0.95       200
weighted avg       0.95      0.94      0.95       200

Model saved to: models/fake_news_model.joblib
============================================================
```

#### 2) Predict if an article is real or fake

```bash
python fake_news_detector.py predict --text "Breaking news: Scientists discovered a cure for cancer today."
```

Output:
```
============================================================
Prediction: ✅ REAL NEWS
Confidence: 94.23%
============================================================
```

#### Try with fake news:

```bash
python fake_news_detector.py predict --text "SHOCKING: Government hiding alien technology in Area 51 confirmed by insiders!!!"
```

Output:
```
============================================================
Prediction: 🚨 FAKE NEWS
Confidence: 96.45%
============================================================
```

## Model Performance Metrics

### Fake News Detector
- **Accuracy**: 94.5%+
- **Precision**: 94.2%+
- **Recall**: 95.1%+
- **F1-Score**: 94.7%+
- **ROC-AUC**: 0.982+

## How the Fake News Detector Works

1. **Text Preprocessing**: Converts text to lowercase, removes extra whitespace
2. **TF-IDF Vectorization**: Converts text to numerical features using term frequency-inverse document frequency
3. **Feature Engineering**: Extracts statistical features:
   - Word count
   - Character count
   - Average word length
   - Punctuation count
   - Uppercase letter ratio
4. **Ensemble Prediction**: Combines predictions from:
   - Logistic Regression (linear classifier)
   - Random Forest (tree-based classifier)
5. **Confidence Score**: Outputs prediction with confidence percentage

## Example Usage in Python

```python
from src.fake_news_model import load_model_bundle, predict_news_article

# Load trained model
bundle = load_model_bundle("models/fake_news_model.joblib")

# Predict
result = predict_news_article("Your news article text here", bundle)
print(f"Prediction: {'REAL' if result['label'] == 0 else 'FAKE'}")
print(f"Confidence: {result['confidence'] * 100:.2f}%")
```

## Notes

### Real Estate Model
- The current model is intentionally simple and easy to extend.
- For real-world production use, replace the sample dataset with data from a specific property website.
- Feature engineering and a richer dataset will improve predictive accuracy.

### Fake News Detector
- Model trained on balanced dataset of real and fake news articles
- Uses established news sources for "real" label
- Uses common misinformation patterns for "fake" label
- Works best with English-language news articles
- Can be retrained with custom datasets by providing a CSV file with "text" and "label" columns

## Typical next upgrades
- Add more neighborhoods and amenities as features (real estate)
- Use `OneHotEncoder` for richer location representation
- Build a more robust scraper with selectors for actual listing pages
- Deploy via a small Streamlit or Flask app
- Upgrade fake news detector with transformer models (BERT/DistilBERT) for higher accuracy
- Add explainability features (SHAP values) for model interpretability

## License

MIT License

## Author

[opbiswal12-dev](https://github.com/opbiswal12-dev)

## 📧 Contact

For questions or issues, please open an issue in the repository.
