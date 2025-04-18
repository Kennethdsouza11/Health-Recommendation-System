import requests
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from dataclasses import dataclass
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for API endpoints and parameters."""
    HF_API_URL: str = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    GKG_API_URL: str = "https://kgsearch.googleapis.com/v1/entities:search"
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 0.5
    TIMEOUT: int = 10
    DEFAULT_LIMIT: int = 20
    DEFAULT_LANGUAGE: str = 'en'
    DEFAULT_THRESHOLD: float = 0.4

class APIClient:
    """Base class for API clients with retry mechanism."""
    def __init__(self, api_key: str, config: APIConfig):
        self.api_key = api_key
        self.config = config
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=config.MAX_RETRIES,
            backoff_factor=config.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

class GoogleKnowledgeGraphClient(APIClient):
    """Client for Google Knowledge Graph API."""
    def get_terms(self, keyword: str, limit: int = None, language: str = None) -> Union[List[str], Dict[str, str]]:
        """
        Retrieve terms from Google's Knowledge Graph API.
        
        Args:
            keyword: Search keyword
            limit: Maximum number of results
            language: Language code
            
        Returns:
            List of terms or error dictionary
        """
        params = {
            'query': keyword,
            'key': self.api_key,
            'limit': limit or self.config.DEFAULT_LIMIT,
            'languages': language or self.config.DEFAULT_LANGUAGE
        }

        try:
            response = self.session.get(
                self.config.GKG_API_URL,
                params=params,
                timeout=self.config.TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            terms = [
                item.get('result', {}).get('name', '')
                for item in data.get('itemListElement', [])
            ]
            return terms
        except requests.exceptions.RequestException as e:
            logger.error(f"GKG API request failed: {str(e)}")
            return {"error": f"API request failed: {str(e)}"}

class HuggingFaceClient(APIClient):
    """Client for Hugging Face Inference API."""
    def get_similarity_scores(self, source: str, targets: List[str]) -> Union[List[float], Dict[str, str]]:
        """
        Get similarity scores between source and target sentences.
        
        Args:
            source: Source sentence
            targets: List of target sentences
            
        Returns:
            List of similarity scores or error dictionary
        """
        payload = {
            "inputs": {
                "source_sentence": source,
                "sentences": targets
            }
        }

        try:
            response = self.session.post(
                self.config.HF_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.config.TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HF API request failed: {str(e)}")
            return {"error": f"API request failed: {str(e)}"}

class TermSimilarityFinder:
    """Main class for finding similar terms using both APIs."""
    def __init__(self):
        load_dotenv()
        
        self.config = APIConfig()
        
        # Initialize API clients
        self.gkg_client = GoogleKnowledgeGraphClient(
            os.getenv('gkg_api'),
            self.config
        )
        self.hf_client = HuggingFaceClient(
            os.getenv('hf_key'),
            self.config
        )
        
        # Validate API keys
        if not self.gkg_client.api_key or not self.hf_client.api_key:
            raise ValueError("Missing API keys. Set 'gkg_api' and 'hf_key' environment variables.")

    @lru_cache(maxsize=100)
    def get_similar_terms(self, keyword: str, threshold: float = None) -> List[str]:
        """
        Find similar terms for a given keyword.
        
        Args:
            keyword: Search keyword
            threshold: Similarity threshold
            
        Returns:
            List of similar terms
        """
        # Get Knowledge Graph terms
        kg_results = self.gkg_client.get_terms(keyword)
        if isinstance(kg_results, dict) and "error" in kg_results:
            logger.error(f"Failed to get KG terms: {kg_results['error']}")
            return []

        # Get similarity scores
        hf_results = self.hf_client.get_similarity_scores(keyword, kg_results)
        if isinstance(hf_results, dict) and "error" in hf_results:
            logger.error(f"Failed to get similarity scores: {hf_results['error']}")
            return []

        # Filter and sort terms
        threshold = threshold or self.config.DEFAULT_THRESHOLD
        similar_terms = [
            {"term": term, "score": score}
            for term, score in zip(kg_results, hf_results)
            if score >= threshold
        ]
        similar_terms.sort(key=lambda x: x["score"], reverse=True)

        return [item["term"] for item in similar_terms]

# Example usage
if __name__ == "__main__":
    try:
        finder = TermSimilarityFinder()
        keyword = "dextrose"
        similar_terms = finder.get_similar_terms(keyword)
        print("Similar Terms:", similar_terms)
    except Exception as e:
        logger.error(f"Error: {str(e)}")