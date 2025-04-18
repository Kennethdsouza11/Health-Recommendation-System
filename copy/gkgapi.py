import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Hugging Face API setup
hf_key = os.getenv('hf_key')
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
hf_headers = {"Authorization": hf_key }

# Google Knowledge Graph API setup
def get_kg_terms_and_categories(keyword, limit=20, language='en'):
    """Retrieve terms and categories from Google's Knowledge Graph API."""
    gkg_api_key = os.getenv('gkg_api')
    if not gkg_api_key:
        raise ValueError("API key not found. Set the 'gkg_api' environment variable.")
    
    gkg_url = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        'query': keyword,
        'key': gkg_api_key,
        'limit': limit,
        'languages': language
    }

    response = requests.get(gkg_url, params=params)
    if response.status_code == 200:
        data = response.json()
        terms = [item.get('result', {}).get('name', '') for item in data.get('itemListElement', [])]
        return terms
    else:
        return {"error": f"API request failed with status code {response.status_code}"}

# Hugging Face Semantic Similarity
def query_hf(payload):
    """Query the Hugging Face Inference API."""
    response = requests.post(HF_API_URL, headers=hf_headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API request failed with status code {response.status_code}"}

# Main Function
def get_similar_terms(keyword, threshold=0.4):
    """Fetch similar terms exceeding the similarity threshold."""
    # Step 1: Get Knowledge Graph terms
    kg_results = get_kg_terms_and_categories(keyword, limit=10)
    
    if isinstance(kg_results, dict) and "error" in kg_results:
        return kg_results  # Return error if KG API fails

    # Step 2: Use Hugging Face API for semantic similarity
    hf_payload = {
        "inputs": {
            "source_sentence": keyword,
            "sentences": kg_results
        }
    }
    hf_results = query_hf(hf_payload)
    
    if isinstance(hf_results, dict) and "error" in hf_results:
        return hf_results  # Return error if HF API fails

    # Step 3: Filter terms based on the threshold and sort by similarity score
    similar_terms = [
        {"term": term, "score": score}
        for term, score in zip(kg_results, hf_results)
        if score >= threshold
    ]
    similar_terms.sort(key=lambda x: x["score"], reverse=True)  # Sort by descending score

    # Step 4: Return filtered terms
    return [item["term"] for item in similar_terms]

# Example Usage
if __name__ == "__main__":
    keyword = "dextrose"
    threshold = 0.4
    similar_terms = get_similar_terms(keyword, threshold)
    
    if isinstance(similar_terms, dict) and "error" in similar_terms:
        print(similar_terms["error"])
    else:
        print("Similar Terms:", similar_terms)