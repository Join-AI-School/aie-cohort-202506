import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Add the project root to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from embeddings import EmbeddingModel
from qdrant import qdrant_client
from qdrant_util import create_collection, upsert_points, search

load_dotenv()

BASE_URL = "https://data.alpaca.markets/v1beta1/news"

def set_headers(api_key, secret_key):
    """Set the headers for the API request."""
    return {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }

def clean_text(content):
    """Clean and transform article content."""
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    
    # Remove all script, style, and iframe elements
    for script_or_style in soup(['script', 'style', 'iframe']):
        script_or_style.decompose()
    
    # Remove all figures and images
    for figure in soup.find_all(['figure', 'img']):
        figure.decompose()
    
    # Extract text from remaining HTML
    text = soup.get_text()
    
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')

    # Remove extra whitespace and newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newlines
    text = re.sub(r' +', ' ', text)  # Replace multiple spaces with single space
    
    # Remove special HTML entities
    text = re.sub(r'&[a-zA-Z0-9]+;', ' ', text)  # Remove HTML entities like &nbsp;
    
    # Remove common disclaimers and boilerplate text
    disclaimers = [
        "Image via Shutterstock",
        "Disclaimer:",
        "This content was partially produced with the help of AI tools",
        "Read More:",
        "See Also:",
        "Read Next:"
    ]
    
    for disclaimer in disclaimers:
        text = re.sub(f".*{re.escape(disclaimer)}.*\n?", "", text)
    
    # Normalize double spacing and paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # No more than double line breaks    
    
    return text.strip()

def extract(ticker, headers, date_range):
    """Extract news articles for a given ticker and date range."""
    url = (
        f"{BASE_URL}?sort=desc"
        f"&start={date_range[0]}"
        f"&end={date_range[1]}"
        f"&symbols={ticker}"
        "&include_content=true"
        "&limit=3"
    )

    response = requests.get(url, headers=headers)
    
    # Parse the JSON response
    if response.status_code == 200:
        return response.json()
    else:            
        print(f"Error: {response.status_code} - {response.text}")
        return None

def transform(ticker, data):
    """Transform the extracted data into a structured format."""
    store = [] 
    articles = data.get('news', [])
    
    for article in articles:
        # Clean content
        content = clean_text(article.get('content', ''))
        # Store the transformed article
        store.append({  
            'ticker': ticker,
            'created_at': article.get('created_at', ''),
            'updated_at': article.get('updated_at', ''),
            'headline': article.get('headline', ''),
            'content': content,
            'url': article.get('url', '')
        })
    return store

def load(embedding_model, articles, collection):
    """Load the transformed articles into the Qdrant database."""
    texts = [article['content'] for article in articles]
    embeddings = embedding_model.encode_texts(texts)    

    # Upsert news articles to the collection 
    upsert_points(qdrant_client, collection, embeddings, articles)

def run_news_etl(tickers, headers, embedding_model):
    """Run the ETL process for news articles."""

    # Get the current date
    current_date = datetime.now()

    # Check if the collection already exists, if not create a new one    
    collection_name = 'news'
    create_collection(qdrant_client, collection_name, vector_size=768)

    # Run ETL for each date and ticker
    for i in range(30):
        end_date = current_date - timedelta(days=i)
        start_date = end_date - timedelta(days=1)
        
        date_range = (start_date.strftime('%Y-%m-%dT00:00:00Z'), end_date.strftime('%Y-%m-%dT00:00:00Z'))
        
        for ticker in tickers:
            # Step 1 - Extraction Data
            raw_data = extract(ticker, headers, date_range)            
            if raw_data:
                # Step 2 - Transform Data
                articles = transform(ticker, raw_data)
                # Step 3 - Load Data
                load(embedding_model, articles, collection_name)

            # Pause for 1 second to prevent API limits
            time.sleep(1)            

# Run the ETL process for the specified tickers
run_news_etl(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"], 
    headers=set_headers(api_key=os.getenv("ALPACA_API_KEY"), secret_key=os.getenv("ALPACA_SECRET_KEY")), 
    embedding_model=EmbeddingModel("./src/embeddings/multi-qa-mpnet-base-dot-v1")
)