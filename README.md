# Health Recommendation System from Packaged Food Ingredients

## Overview

This project allows users to upload images of packaged foods, which are then processed using Optical Character Recognition (OCR) to extract the text from the image. The extracted text is then passed through a keyword extraction algorithm (YAKE), which extracts relevant keywords. These keywords are then used to fetch relevant articles from the arXiv API and Wikipedia, which are combined to generate personalized health recommendations using the Llama 3.3 70B LLM (Large Language Model).

The system aims to help users understand the health implications of the ingredients present in packaged food and provide health recommendations based on the extracted information.

## Features

- **Image Upload**: Users can upload images of packaged food.
- **OCR Text Extraction**: The system uses Optical Character Recognition (OCR) to extract text from the uploaded images.
- **Keyword Extraction with YAKE**: YAKE (Yet Another Keyword Extractor) is used to extract relevant keywords from the extracted text.
- **Article Extraction**: Relevant articles are fetched from the arXiv API and Wikipedia using the extracted keywords.
- **Personalized Health Recommendations**: The information from these articles is processed and used by the Llama 3.3 70B model to generate personalized health recommendations.

## Technologies Used

- **OCR**: For text extraction from images (e.g., Tesseract OCR or other OCR libraries).
- **YAKE**: For keyword extraction from the text.
- **arXiv API**: For fetching relevant scientific articles related to the extracted keywords.
- **Wikipedia API**: For fetching relevant articles from Wikipedia.
- **Llama 3.3 70B**: A large language model (LLM) used to generate health recommendations based on the extracted knowledge.
- **Python**: The primary programming language used for the project.
- **Streamlit**: For creating the web interface for users to interact with the application.

## Usage

1. Open the application in your browser by navigating to the provided local URL (usually `http://localhost:8501`).
2. Upload an image of a packaged food item using the file uploader.
3. The system will extract the text from the image using OCR, extract keywords using YAKE, fetch relevant articles from arXiv and Wikipedia, and process the information using the Llama 3.3 70B model.
4. The personalized health recommendations will be displayed on the screen.

## Example Flow

1. **Upload Image**: A user uploads an image of a food package, such as a box of cereal or a snack bar.
2. **OCR Process**: The system extracts the text from the image, such as the ingredients list and nutritional information.
3. **Keyword Extraction**: Keywords like "sugar," "fiber," "vitamins," etc., are extracted from the text.
4. **Article Extraction**: The system uses these keywords to query the arXiv and Wikipedia APIs for relevant scientific and health articles.
5. **Personalized Recommendation**: The extracted knowledge is processed by the Llama 3.3 70B model to generate a personalized health recommendation for the user.

