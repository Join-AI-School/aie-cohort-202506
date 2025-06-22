import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# Load environment variables from .env file
# ------------------------------------------------------------------------------
load_dotenv()

# Extract Qdrant API key from environment variable
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = "https://9ea54df5-9373-4356-804d-1d0cb5731bcb.us-east4-0.gcp.cloud.qdrant.io"

# ------------------------------------------------------------------------------
# Create a Qdrant client instance
# ------------------------------------------------------------------------------

qdrant_client = QdrantClient(
    url=QDRANT_ENDPOINT, 
    api_key=QDRANT_API_KEY,  # Auth required for Qdrant Cloud
)

# ------------------------------------------------------------------------------
# Step 2: Initialize Embeddings
# ------------------------------------------------------------------------------

embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

# ------------------------------------------------------------------------------
# Search Function
# ------------------------------------------------------------------------------

def vector_search(query, limit_size=3):
    query_embedding = embedding_model.encode(query).tolist()
    search_results = qdrant_client.search(
        collection_name="news",
        query_vector=query_embedding,
        limit=limit_size,
    )
    return search_results