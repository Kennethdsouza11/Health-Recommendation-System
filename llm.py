import streamlit as st
from groq import Groq
from paddle_ocr import OCRProcessor
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from preprocess import KeywordExtractor
from PIL import Image
import io
from context_FoodDataCentral import fetch_food_context
from context import fetch_context
from typing import Dict, Optional, Any
import logging
import tempfile
from pathlib import Path
import shutil
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Configuration for the application."""
    MAX_FILE_SIZE_MB: int = 5
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    TEMP_DIR: str = "temp"
    MODEL_NAME: str = "llama-3.3-70b-versatile"
    COMPRESSION_QUALITY: int = 70
    MAX_CONCURRENT_REQUESTS: int = 5

class HealthRecommendationSystem:
    """Main class for the health recommendation system."""
    def __init__(self):
        load_dotenv()
        self.config = AppConfig()
        self.api_key = os.getenv('API_KEY')
        if not self.api_key:
            raise ValueError("API key is missing. Please set the API_KEY in your .env file.")
        
        self.client = Groq(api_key=self.api_key)
        self.ocr_processor = OCRProcessor()
        self.keyword_extractor = KeywordExtractor()
        self._setup_temp_dir()
        self._setup_ui()

    def _setup_temp_dir(self):
        """Create and setup temporary directory."""
        self.temp_dir = Path(self.config.TEMP_DIR)
        self.temp_dir.mkdir(exist_ok=True)

    def _setup_ui(self):
        """Setup Streamlit UI components."""
        st.markdown("""
            <style>
                .stApp {
                    background: url('https://cdn.expresshealthcare.in/wp-content/uploads/2019/08/21181504/GettyImages-1040917000-1-e1566391534758-750x357.jpg') no-repeat center center fixed;
                    background-size: cover;
                    opacity: 1;
                }
                .main {
                    background-color: rgba(255, 255, 255, 0.85);
                    padding: 20px;
                    border-radius: 10px;
                }
                .stButton>button {
                    background-color: #008CBA;
                    color: white;
                    border-radius: 5px;
                }
                .stTextInput, .stTextArea, .stNumberInput, .stSelectbox {
                    border-radius: 5px;
                }
            </style>
        """, unsafe_allow_html=True)

        st.sidebar.title("Navigation")
        st.sidebar.markdown("""
            - 📄 **Enter Patient Details**
            - 🖼️ **Upload Image for Analysis**
            - 🔍 **Get Personalized Insights**
        """)

    def _validate_user_details(self, details: Dict[str, Any]) -> bool:
        """Validate user input details."""
        required_fields = ["name", "age", "gender", "weight", "height"]
        return all(details.get(field) for field in required_fields)

    def enter_details(self) -> Optional[Dict[str, Any]]:
        """Collect and validate user details."""
        st.header("🩺 Patient Information")
        with st.expander("Fill in your details 👇", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name:")
                age = st.number_input("Age:", min_value=0, max_value=120, step=1)
                gender = st.radio("Gender:", ["Male", "Female"], horizontal=True)
            with col2:
                phone_number = st.text_input("Phone Number:")
            
            weight = st.number_input("Weight (kg):", min_value=0.0, max_value=500.0, step=0.1)
            height = st.number_input("Height (cm):", min_value=0.0, max_value=300.0, step=0.1)
            allergies = st.text_area("Allergies:")
            medications = st.text_area("Ongoing Medications:")
            conditions = st.text_area("Medical Conditions:")

            if st.button("✅ Submit Details"):
                details = {
                    "name": name, "age": age, "gender": gender,
                    "phone": phone_number, "weight": weight,
                    "height": height, "allergies": allergies,
                    "medications": medications, "conditions": conditions
                }
                if self._validate_user_details(details):
                    return details
                else:
                    st.error("Please fill in all required fields.")
        return None

    def process_uploaded_image(self, uploaded_file) -> Optional[str]:
        """Handle image processing with proper cleanup."""
        if uploaded_file.size > self.config.MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File too large! Max size: {self.config.MAX_FILE_SIZE_MB}MB.")
            return None

        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=self.temp_dir) as temp_file:
                image = Image.open(uploaded_file)
                image.save(temp_file.name, "JPEG", optimize=True, quality=self.config.COMPRESSION_QUALITY)
                return temp_file.name
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            st.error("Image processing failed. Please try again.")
            return None

    def _cleanup_temp_files(self):
        """Cleanup temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Failed to cleanup temp directory: {str(e)}")

    def analyze_image(self, image_path: str, user_details: Dict[str, Any]):
        """Perform analysis with proper error handling and retries."""
        try:
            with st.spinner("🔍 Analyzing Image..."):
                # Perform OCR
                ocr_text = self.ocr_processor.extract_text_from_image(image_path)
                if not ocr_text:
                    st.error("Failed to extract text from image.")
                    return

                # Extract keywords
                keys = self.keyword_extractor.extract_keywords(ocr_text)
                if not keys:
                    st.error("Failed to extract keywords.")
                    return

                # Fetch context with retries
                for attempt in range(self.config.MAX_RETRIES):
                    try:
                        food_context = fetch_food_context(keys)
                        additional_context = fetch_context(keys)
                        break
                    except Exception as e:
                        if attempt == self.config.MAX_RETRIES - 1:
                            raise
                        time.sleep(self.config.RETRY_DELAY)

                combined_context = food_context + "\n\n" + additional_context

                # Generate summary
                summary = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Summarize the context."},
                        {"role": "user", "content": combined_context}
                    ],
                    model=self.config.MODEL_NAME
                )

                # Generate analysis
                analysis_prompt = f"Analyze {ocr_text} based on {user_details}. Provide health insights."
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": summary.choices[0].message.content},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    model=self.config.MODEL_NAME
                )

                st.success("✅ Analysis Complete!")
                st.subheader("Personalized Insights 🧑‍⚕️")
                st.write(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            st.error("Analysis failed. Please try again.")

    def run(self):
        """Main application loop."""
        try:
            st.title("🏥 Health Recommendation System")
            uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"], help=f"Max: {self.config.MAX_FILE_SIZE_MB}MB")
            
            user_details = self.enter_details()
            if uploaded_file and user_details:
                image_path = self.process_uploaded_image(uploaded_file)
                if image_path:
                    self.analyze_image(image_path, user_details)
                else:
                    st.error("❌ Image processing failed.")
            else:
                st.warning("Upload an image and enter details to proceed.")
        finally:
            self._cleanup_temp_files()

if __name__ == "__main__":
    try:
        app = HealthRecommendationSystem()
        app.run()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        st.error("Application failed to start. Please check the logs for details.")
