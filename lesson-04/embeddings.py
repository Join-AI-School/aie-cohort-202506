import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class EmbeddingModel(Embeddings):
    def __init__(
        self,
        local_model_path: str = "./models/multi-qa-mpnet-base-dot-v1",
        remote_model_name: str = "multi-qa-mpnet-base-dot-v1",
    ):
        """
        :param local_model_path: Where to look for (and possibly save) the model.
        :param remote_model_name: Name of the model on Hugging Face.
        """
        # Check if the local path exists
        if os.path.isdir(local_model_path):
            # If it exists, load from local
            print(f"Loading model from local path: {local_model_path}")
            self.model = SentenceTransformer(local_model_path)
        else:
            # Otherwise, download from remote and then save to local
            print(f"Local model not found. Downloading '{remote_model_name}'...")
            self.model = SentenceTransformer(remote_model_name)
            # Create parent directories if needed, then save the model
            os.makedirs(local_model_path, exist_ok=True)
            self.model.save(local_model_path)
            print(f"Model saved locally to: {local_model_path}")

    def embed_documents(self, texts):
        """
        Embed a list of documents.
        
        :param texts: List of texts to embed
        :return: List of embeddings
        """
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        """
        Embed a single query text.
        
        :param text: Text to embed
        :return: Embedding for the text
        """
        return self.model.encode(text).tolist()

    # Optional: Keep your original method if needed
    def encode_texts(self, texts):
        return self.model.encode(texts, show_progress_bar=True)
