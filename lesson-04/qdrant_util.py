import hashlib
import logging
from time import time
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct
)

logger = logging.getLogger(__name__)

def create_collection(client, name, vector_size=768, distance=Distance.COSINE):
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=distance),
    )

def unique_id_generator(payload):
    payload = payload.get('metadata')
    unique_id = int(hashlib.md5(f"{payload['ticker']}_{payload.get('created_at')}__{time() * 1000}".encode()).hexdigest(), 16) % (10 ** 10)
    return unique_id

def create_news_payload(article):
    return {
        "metadata": {
            "ticker": article.get("ticker"),
            'created_at': article.get('created_at', ''),
            'updated_at': article.get('updated_at', ''),
            'headline': article.get('headline', ''),
            'url': article.get('url','')
        },
        "page_content": article.get("content")
    }

def create_earnings_payload(filing):
    return {
        "metadata": {
            "ticker": filing.get("ticker"),
            'created_at': filing.get('created_at', ''),            
            'year': filing.get('year', ''),            
            'url': filing.get('url','')
        },
        "page_content": filing.get("content")
    }

def set_router(collection):
    router = {
        'news': create_news_payload,
        'earnings': create_earnings_payload
    }

    return router[collection]

def upsert_points(client, collection_name, embeddings, items, batch_size=100):

    points = []
    for idx, (item, embedding) in enumerate(zip(items, embeddings)):

        try:            
            vector = embedding.tolist()  # Convert embedding to a list

            # Prepare payload in LangChain Document format
            create_payload = set_router(collection_name)
            payload = create_payload(item)

            # Generate unique ID
            idx = unique_id_generator(payload)

            # Log data for debugging
            logger.debug(f"Point ID: {idx}, Vector Length: {len(vector)}, Payload Keys: {payload.keys()}")

            point = PointStruct(
                id=idx,
                vector=vector,
                payload=payload
            )
            points.append(point)

        except Exception as e:
            logger.error(f"Error creating point at index {idx}: {e}")
            continue  # Skip to the next point

        # Upsert in batches
        if len(points) >= batch_size:
            try:
                client.upsert(collection_name=collection_name, points=points, wait=True)
                logger.info(f"Upserted {len(points)} points.")
                points = []  # Clear the batch
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")
                break  # Stop upserting if a batch fails

    # Upsert any remaining points
    if points:
        try:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            logger.info(f"Upserted remaining {len(points)} points.")
        except Exception as e:
            logger.error(f"Error upserting remaining batch: {e}")

def search(client, collection_name, query_vector, limit=3):
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit
    )
    return results
