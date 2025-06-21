import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# ------------------------------------------------------------------------------
# Configure logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # You can set DEBUG, INFO, WARNING, etc.
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Load environment variables from .env file
# ------------------------------------------------------------------------------
load_dotenv()

# Extract Qdrant API key from environment variable
api_key = os.getenv("QDRANT_API_KEY")
logger.info(f"QDRANT_API_KEY: {os.getenv('QDRANT_API_KEY')}")

# ------------------------------------------------------------------------------
# Create a Qdrant client instance
# ------------------------------------------------------------------------------
qdrant_url = os.getenv("QDRANT_ENDPOINT")
logger.info(f"Qdrant URL: {qdrant_url}")

qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=api_key,  # Auth required for Qdrant Cloud
)

# ------------------------------------------------------------------------------
# Attempt to retrieve the list of collections
# ------------------------------------------------------------------------------
try:
    response = qdrant_client.get_collections()
    # If successful, log the response
    logger.info("Qdrant Connection successful. Collections retrieved:")
    logger.info(response)
except Exception as e:
    # Log the error if something goes wrong
    logger.error(f"Connection failed: {e}")
