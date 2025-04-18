import yake
from typing import List, Optional
import logging
from dataclasses import dataclass
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class YAKEConfig:
    """Configuration for YAKE keyword extraction."""
    DEFAULT_LANGUAGE: str = "en"
    MAX_NGRAM_SIZE: int = 3
    DEDUP_THRESHOLD: float = 0.9
    DEDUP_ALGO: str = 'seqm'
    WINDOW_SIZE: int = 1
    DEFAULT_TOP_N: int = 10
    CACHE_SIZE: int = 100  # Number of results to cache

class KeywordExtractor:
    """Keyword extractor with caching and error handling."""
    def __init__(self, config: Optional[YAKEConfig] = None):
        self.config = config or YAKEConfig()
        self.extractor = self._initialize_extractor()

    def _initialize_extractor(self) -> yake.KeywordExtractor:
        """Initialize YAKE keyword extractor."""
        try:
            return yake.KeywordExtractor(
                lan=self.config.DEFAULT_LANGUAGE,
                n=self.config.MAX_NGRAM_SIZE,
                dedupLim=self.config.DEDUP_THRESHOLD,
                dedupFunc=self.config.DEDUP_ALGO,
                windowsSize=self.config.WINDOW_SIZE
            )
        except Exception as e:
            logger.error(f"Failed to initialize YAKE: {str(e)}")
            raise

    def _validate_text(self, text: str) -> bool:
        """Validate input text."""
        if not text or not isinstance(text, str):
            logger.error("Invalid text input")
            return False
        if len(text.strip()) == 0:
            logger.warning("Empty text input")
            return False
        return True

    @lru_cache(maxsize=YAKEConfig.CACHE_SIZE)
    def extract_keywords(self, text: str, top_n: Optional[int] = None) -> List[str]:
        """
        Extract keywords from text using YAKE.
        
        Args:
            text: Input text to process.
            top_n: Number of keywords to extract (defaults to config.DEFAULT_TOP_N).
            
        Returns:
            List of extracted keywords.
        """
        try:
            # Validate input
            if not self._validate_text(text):
                return []

            # Extract keywords
            keywords = self.extractor.extract_keywords(text)
            
            # Handle empty results
            if not keywords:
                logger.warning("No keywords extracted")
                return []

            # Return top_n keywords
            top_n = top_n or self.config.DEFAULT_TOP_N
            top_keywords = [kw[0] for kw in keywords[:top_n]]
            
            return top_keywords

        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []

    def batch_extract(self, texts: List[str], top_n: Optional[int] = None) -> List[List[str]]:
        """
        Extract keywords from multiple texts in batch.
        
        Args:
            texts: List of texts to process.
            top_n: Number of keywords to extract per text.
            
        Returns:
            List of keyword lists (empty list for failed extractions).
        """
        results = []
        for text in texts:
            try:
                keywords = self.extract_keywords(text, top_n)
                results.append(keywords)
            except Exception as e:
                logger.error(f"Batch extraction failed for text: {str(e)}")
                results.append([])
        return results

# Example usage
if __name__ == "__main__":
    try:
        # Initialize extractor
        extractor = KeywordExtractor()
        
        # Process single text
        text = "This is an example text for keyword extraction."
        keywords = extractor.extract_keywords(text)
        print(f"Extracted keywords: {keywords}")
        
        # Process multiple texts
        texts = [
            "First example text.",
            "Second example text."
        ]
        results = extractor.batch_extract(texts)
        for text, keywords in zip(texts, results):
            print(f"{text}: {keywords}")
            
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")






