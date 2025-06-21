"""
Lesson 04: Qdrant
-----------------------------------------

Qdrant is a vector database that can be used to store and query 
embeddings efficiently. LangChain integrates with Qdrant to enable 
seamless storage and retrieval of vectorized data.

Make sure you have the necessary libraries installed:

```
uv add qdrant-client sentence-transformers
```

"""
#%%
import os
from warnings import filterwarnings
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Load environment variables from .env file
# ------------------------------------------------------------------------------

# Extract Qdrant API key from environment variable
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")

#%%
# ------------------------------------------------------------------------------
# Step 1: Connect to Qdrant
# ------------------------------------------------------------------------------

qdrant_client = QdrantClient(
    url=QDRANT_ENDPOINT, 
    api_key=QDRANT_API_KEY,  # Auth required for Qdrant Cloud
)

#%% 
# ------------------------------------------------------------------------------
# Step 2: Initialize Embeddings
# ------------------------------------------------------------------------------

embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Sample data to ingest
import uuid

documents = [
    {"id": str(uuid.uuid4()), "content": "Qdrant is a vector database for efficient similarity search."},
    {"id": str(uuid.uuid4()), "content": "LangChain integrates with Qdrant for seamless vector storage."},
    {"id": str(uuid.uuid4()), "content": "HuggingFace provides state-of-the-art embedding models."},
]

# Generate embeddings for the documents
for doc in documents:
    doc["embedding"] = embedding_model.encode(doc["content"])

#%%
# ------------------------------------------------------------------------------
# Step 3: Create Collection
# ------------------------------------------------------------------------------

# Create a collection in Qdrant and upload the documents
qdrant_client.recreate_collection(
    collection_name="sample_collection",
    vectors_config={"size": len(documents[0]["embedding"]), "distance": "Cosine"},
)

#%%
# ------------------------------------------------------------------------------
# Upload Collection
# ------------------------------------------------------------------------------
qdrant_client.upsert(
    collection_name="sample_collection",
    points=[
        PointStruct(
            id=doc["id"],
            vector=doc["embedding"].tolist(),
            payload={"content": doc["content"]}
        )
        for doc in documents
    ]
)

# ------------------------------------------------------------------------------
# Upload Collection
# ------------------------------------------------------------------------------

# Query the collection
query = "LangChain?"
query_embedding = embedding_model.encode(query).tolist()
search_results = qdrant_client.search(
    collection_name="sample_collection",
    query_vector=query_embedding,
    limit=3,
)

# Print the retrieved document
for result in search_results:
    print(f"Retrieved Document: {result.payload['content']}")
