from concurrent.futures import ThreadPoolExecutor
from langchain_community.retrievers import ArxivRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from functools import lru_cache
import threading
from typing import List, Dict, Optional
import time
from cachetools import TTLCache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    TOKEN_LIMIT = 200
    CACHE_SIZE = 1000
    CACHE_TTL = 3600  # 1 hour
    MAX_PAGES = 2
    COSINE_THRESHOLD = 0.2
    MAX_WORKERS = 5

# Thread-safe cache with TTL
cache = TTLCache(maxsize=Config.CACHE_SIZE, ttl=Config.CACHE_TTL)
cache_lock = threading.Lock()

# Lock for synchronizing file access
file_lock = threading.Lock()

# Global vectorizer instance
vectorizer = TfidfVectorizer()

@lru_cache(maxsize=100)
def compute_cosine_similarity(key: str, doc_content: str) -> float:
    """
    Compute Cosine Similarity between two texts using TF-IDF.
    
    Args:
        key: The search key
        doc_content: The document content to compare
        
    Returns:
        float: Similarity score between 0 and 1
    """
    try:
        tfidf_matrix = vectorizer.fit_transform([key, doc_content])
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity_matrix[0, 0]
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        return 0.0

def truncate_text(text: str, token_limit: int) -> str:
    """
    Truncate text to a specific token limit.
    
    Args:
        text: The text to truncate
        token_limit: Maximum number of tokens allowed
        
    Returns:
        str: Truncated text
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) > token_limit:
            tokens = tokens[:token_limit]
        return encoding.decode(tokens)
    except Exception as e:
        logger.error(f"Error truncating text: {str(e)}")
        return text[:1000]  # Fallback to simple truncation

class ArxivContextRetriever:
    def __init__(self, max_pages: int = Config.MAX_PAGES):
        self.max_pages = max_pages
        self.retriever = ArxivRetriever(load_max_docs=max_pages, get_full_documents=True)
        
    def fetch_documents(self, key: str) -> List[str]:
        """
        Fetch documents from Arxiv for a given key.
        
        Args:
            key: The search key
            
        Returns:
            List[str]: List of document contents
        """
        try:
            with file_lock:
                documents = self.retriever.invoke(key)[:self.max_pages]
            return [doc.page_content for doc in documents]
        except Exception as e:
            logger.error(f"Error fetching documents for key '{key}': {str(e)}")
            return []

def fetch_context_for_key(key: str, retriever: ArxivContextRetriever) -> str:
    """
    Fetch relevant context for a single key based on cosine similarity.
    
    Args:
        key: The search key
        retriever: ArxivContextRetriever instance
        
    Returns:
        str: Combined and truncated context
    """
    # Check cache first
    with cache_lock:
        if key in cache:
            return cache[key]
            
    documents = retriever.fetch_documents(key)
    filtered_context = []

    for doc_content in documents:
        similarity = compute_cosine_similarity(key, doc_content)
        if similarity >= Config.COSINE_THRESHOLD:
            filtered_context.append(doc_content[:1000])

    combined_context = "\n\n".join(filtered_context)
    truncated_context = truncate_text(combined_context, Config.TOKEN_LIMIT)

    # Update cache
    with cache_lock:
        cache[key] = truncated_context
        
    return truncated_context

def fetch_context(keys: List[str]) -> str:
    """
    Fetch relevant context for multiple keys in parallel.
    
    Args:
        keys: List of search keys
        
    Returns:
        str: Combined context for all keys
    """
    retriever = ArxivContextRetriever()
    
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        results = list(
            executor.map(
                lambda key: fetch_context_for_key(key, retriever),
                keys
            )
        )
    
    return "\n\n---\n\n".join(results)
