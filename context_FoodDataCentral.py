import os
import concurrent.futures
import requests
import tiktoken
import logging
import time
from typing import List, Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodDataAPI:
    def __init__(self, api_key: str, max_retries: int = 3, backoff_factor: float = 0.5):
        self.api_key = api_key
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_food_data(self, key: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """
        Fetch food data for a given key with retry mechanism.
        
        Args:
            key: The ingredient name to search for
            timeout: Request timeout in seconds
            
        Returns:
            Optional[Dict]: JSON response from API or None if request fails
        """
        url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            "query": key,
            "pageSize": 1,
            "api_key": self.api_key,
        }
        
        try:
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for key '{key}': {str(e)}")
            return None

def fetch_food_context(
    keys: List[str],
    max_tokens: int = 128000,
    max_workers: int = 5,
    timeout: int = 5,
    max_retries: int = 3,
    backoff_factor: float = 0.5
) -> str:
    """
    Fetches ingredient data from FoodData Central API using multithreading and ensures the
    combined context does not exceed the specified token limit.

    Args:
        keys: List of ingredient names to search for
        max_tokens: Maximum number of tokens allowed in the combined context
        max_workers: Number of threads to use for concurrent processing
        timeout: Timeout for each HTTP request in seconds
        max_retries: Maximum number of retry attempts for failed requests
        backoff_factor: Backoff factor for retry attempts

    Returns:
        str: Combined context with summaries of the ingredients

    Raises:
        ValueError: If API key is not found in environment variables
    """
    api_key = os.getenv("FOODDATA_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set the FOODDATA_API_KEY environment variable.")

    api = FoodDataAPI(api_key, max_retries, backoff_factor)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    summaries = []
    current_tokens = 0

    def process_food_data(key: str) -> Optional[str]:
        data = api.get_food_data(key, timeout)
        if not data or "foods" not in data or not data["foods"]:
            return None

        food = data["foods"][0]
        description = food.get("description", "")
        brand = food.get("brandOwner", "Unknown brand")
        nutrients = food.get("foodNutrients", [])

        nutrient_summary = ", ".join(
            f"{nutrient['nutrientName']}: {nutrient['value']} {nutrient['unitName']}"
            for nutrient in nutrients[:3]
        )

        return f"{description} (Brand: {brand}). Key nutrients: {nutrient_summary}."

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(process_food_data, key): key for key in keys}
        
        for future in concurrent.futures.as_completed(future_to_key):
            try:
                summary = future.result()
                if summary:
                    # Check token count before adding
                    new_tokens = len(tokenizer.encode(summary))
                    if current_tokens + new_tokens <= max_tokens:
                        summaries.append(summary)
                        current_tokens += new_tokens
                    else:
                        logger.warning(f"Token limit reached. Skipping remaining items.")
                        break
            except Exception as e:
                key = future_to_key[future]
                logger.error(f"Error processing key '{key}': {str(e)}")

    return " ".join(summaries)