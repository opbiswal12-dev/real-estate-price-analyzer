# Real Estate Price Analyzer

A Python project that estimates property prices using a multiple linear regression model trained on structured housing data, with a web-scraping module for collecting listing data from public real estate pages.

## What it does
- Scrapes listing data such as price, location, size, bedrooms, bathrooms, and amenities
- Cleans and preprocesses the dataset
- Trains a multiple regression model to predict house prices
- Prints model metrics and supports single-property price prediction

## Project structure

```text
real-estate-price-analyzer/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   └── sample_houses.csv
├── src/
│   ├── __init__.py
│   ├── scraper.py
│   ├── data_processor.py
│   └── model.py
└── models/
    └── price_model.pkl
```

## Tech stack
- Python 3.10+
- pandas
- scikit-learn
- BeautifulSoup4
- requests
- NumPy

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# or .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Usage

### 1) Train the model

```bash
python main.py train
```

This loads the sample dataset in `data/sample_houses.csv`, trains the regression model, and prints evaluation metrics like R², MAE, and RMSE.

### 2) Scrape listings

```bash
python main.py scrape --url "https://example.com/properties"
```

The scraper attempts to parse real estate listing cards from a page, saves the output to `data/scraped_houses.csv`, and falls back to a generated sample dataset if nothing usable is found.

### 3) Predict a price

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

## Notes
- The current model is intentionally simple and easy to extend.
- For real-world production use, you should replace the sample dataset with data from a specific property website and validate the scraped field names against the target site.
- Feature engineering and a richer dataset will improve predictive accuracy.

## Typical next upgrades
- Add more neighborhoods and amenities as features
- Use `OneHotEncoder` for richer location representation
- Build a more robust scraper with selectors for actual listing pages
- Deploy via a small Streamlit or Flask app
