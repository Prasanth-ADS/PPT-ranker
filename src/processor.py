from sentence_transformers import SentenceTransformer
import numpy as np

# Load local model (cached)
# Using 'all-MiniLM-L6-v2' as it is fast and effective
try:
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load embedding model. {e}")
    model = None

def get_embedding(text):
    if not model or not text:
        return np.zeros(384) # Default dimension for MiniLM-L6-v2
    
    # Truncate if too long? Model handles it but let's be safe
    # Actually SentenceTransformer handles truncation usually.
    return model.encode(text)

def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Simple character-based chunking.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def calculate_similarity(emb1, emb2):
    """
    Cosine similarity.
    """
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(emb1, emb2) / (norm1 * norm2)
