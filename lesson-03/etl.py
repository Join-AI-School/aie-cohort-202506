"""
# Libraries to install
uv add google-cloud-bigquery

# Run this in shell before running python etl.py
gcloud auth application-default login

"""
import os
import json
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

# Load environment variables from .env file
load_dotenv()

# Environment Variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")  # e.g., "my-gcp-project"
GCP_BQ_DATASET = os.getenv("GCP_BQ_DATASET")  # default dataset name
GCP_BQ_TABLE = os.getenv("GCP_BQ_TABLE")  # default table name 

# Alpaca Market Data API URL for daily bars (v2)
BASE_URL = "https://data.alpaca.markets/v2/stocks"

def set_headers(api_key, secret_key):
    """Set headers for Alpaca API requests."""
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }

def extract_daily_stock_prices(ticker, start_date, end_date, headers):
    """
    Extract daily stock bars for a given ticker between start_date and end_date.
    """
    url = f"{BASE_URL}/{ticker}/bars?timeframe=1Day&start={start_date}&end={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching {ticker}: {response.status_code} - {response.text}")
        return None
    
def transform_data(ticker, raw_data):
    """
    Transform the raw JSON data into a list of dictionaries matching our schema.
    Expected raw_data structure:
      {
          'bars': [{'c': ..., 'h': ..., 'l': ..., 'n': ..., 'o': ..., 't': ..., 'v': ..., 'vw': ...}],
          'next_page_token': ...,
          'symbol': 'AAPL'
      }
    """
    transformed = []
    for bar in raw_data.get('bars', []):
        transformed.append({
            "ticker": ticker,
            "date": bar.get("t"),             # Timestamp (e.g., "2025-03-14T04:00:00Z")
            "open": bar.get("o"),
            "high": bar.get("h"),
            "low": bar.get("l"),
            "close": bar.get("c"),
            "volume": bar.get("v"),
            "volume_weighted": bar.get("vw"),
            "num_trades": bar.get("n")
        })
    return transformed


# transform_data('MSFT', response)

#%%

def create_dataset_if_not_exists(client, dataset_id):
    """
    Create a BigQuery dataset if it doesn't exist.
    dataset_id should be in the format "project_id.dataset_id"
    """
    try:
        client.get_dataset(dataset_id)
        print(f"Dataset {dataset_id} already exists.")
    except NotFound:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"  # Update as needed
        dataset = client.create_dataset(dataset)
        print(f"Created dataset {dataset_id}.")

def create_table_if_not_exists(project, dataset, table):
    """
    Create the BigQuery table if it doesn't exist.
    """
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"
    schema = [
        bigquery.SchemaField("ticker", "STRING"),
        bigquery.SchemaField("date", "TIMESTAMP"),
        bigquery.SchemaField("open", "FLOAT"),
        bigquery.SchemaField("high", "FLOAT"),
        bigquery.SchemaField("low", "FLOAT"),
        bigquery.SchemaField("close", "FLOAT"),
        bigquery.SchemaField("volume", "INTEGER"),
        bigquery.SchemaField("volume_weighted", "FLOAT"),
        bigquery.SchemaField("num_trades", "INTEGER"),
    ]
    dataset_ref = client.dataset(dataset)
    try:
        client.get_table(table_id)  # Check if table exists
        print(f"Table {table_id} already exists.")
    except NotFound:
        table_obj = bigquery.Table(table_id, schema=schema)
        table_obj = client.create_table(table_obj)
        print(f"Created table {table_id}.")

def load_to_bigquery(rows, project, dataset, table):
    """
    Load the list of rows into a BigQuery table.
    """
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"
    
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        print("Errors while inserting rows:", errors)
    else:
        print("Rows successfully inserted into BigQuery.")

def run_etl(tickers, headers):
    """
    Run the ETL process:
      1. Extract: Retrieve daily stock prices for the past 90 days.
      2. Transform: Format data for BigQuery.
      3. Load: Insert rows into BigQuery.
    """
    # Calculate date range for past 90 days (UTC)
    today = datetime.utcnow()
    start_date = (today - timedelta(days=90)).strftime("%Y-%m-%dT00:00:00Z")
    end_date = today.strftime("%Y-%m-%dT00:00:00Z")

    all_rows = []
    for ticker in tickers:
        print(f"Processing {ticker} from {start_date} to {end_date}...")
        raw_data = extract_daily_stock_prices(ticker, start_date, end_date, headers)
        if raw_data:
            rows = transform_data(ticker, raw_data)
            all_rows.extend(rows)
        time.sleep(1)  # Pause to respect API rate limits

    # Initialize BigQuery client
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    dataset_id = f"{GCP_PROJECT_ID}.{GCP_BQ_DATASET}"
    # Create dataset if it doesn't exist
    create_dataset_if_not_exists(bq_client, dataset_id)
    # Create table if it does not exist
    create_table_if_not_exists(GCP_PROJECT_ID, GCP_BQ_DATASET, GCP_BQ_TABLE)
    # Load transformed rows into BigQuery
    load_to_bigquery(all_rows, GCP_PROJECT_ID, GCP_BQ_DATASET, GCP_BQ_TABLE)

def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
    headers = set_headers(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    run_etl(tickers, headers)

if __name__ == "__main__":
    main()