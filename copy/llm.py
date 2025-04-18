import streamlit as st
from groq import Groq
from paddle_ocr import extract_text_from_image
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from preprocess import yake_keywords
from PIL import Image
import io
from context_FoodDataCentral import fetch_food_context
from context import fetch_context

# Load environment variables
load_dotenv()
key = os.getenv('API_KEY')
if not key:
    st.error("API key is missing. Please set the API_KEY in your .env file.")
    st.stop()

client = Groq(api_key=key)

# Maximum file size for upload (in MB)
MAX_FILE_SIZE_MB = 5

# Custom CSS for transparent background image
st.markdown("""
    <style>
        /* Transparent background image spanning the full page */
        .stApp {
            background: url('https://cdn.expresshealthcare.in/wp-content/uploads/2019/08/21181504/GettyImages-1040917000-1-e1566391534758-750x357.jpg') no-repeat center center fixed;
            background-size: cover;
            opacity: 1; /* Adjust transparency */
        }
        .main {
            background-color: rgba(255, 255, 255, 0.85); /* Slightly opaque white background */
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

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("""
    - 📄 **Enter Patient Details**
    - 🖼️ **Upload Image for Analysis**
    - 🔍 **Get Personalized Insights**
""")

def enter_details():
    """Collect user details with better UI design."""
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
            return {
                "name": name, "age": age, "gender": gender, "phone": phone_number, "email": None,
                "weight": weight, "height": height, "allergies": allergies,
                "medications": medications, "conditions": conditions
            }
    return None

def process_uploaded_image(uploaded_file):
    """Handle image conversion and compression."""
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large! Max size: {MAX_FILE_SIZE_MB}MB.")
        return None

    try:
        image = Image.open(uploaded_file)
        compressed_path = f"compressed_{uploaded_file.name}"
        image.save(compressed_path, "JPEG", optimize=True, quality=70)
        return compressed_path
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        return None

def analyze_image(image_path, user_details):
    """Perform OCR, fetch context, and generate response."""
    with st.spinner("🔍 Analyzing Image..."):
        ocr_text = extract_text_from_image(image_path)
        keys = yake_keywords(ocr_text)
        food_context = fetch_food_context(keys)
        additional_context = fetch_context(keys)
        combined_context = food_context + "\n\n" + additional_context
        
        summary = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Summarize the context."},
                {"role": "user", "content": f"{combined_context}"}
            ],
            model="llama-3.3-70b-versatile"
        )
        
        analysis_prompt = f"Analyze {ocr_text} based on {user_details}. Provide health insights."
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": summary.choices[0].message.content},
                {"role": "user", "content": analysis_prompt}
            ],
            model="llama-3.3-70b-versatile"
        )
        
    st.success("✅ Analysis Complete!")
    st.subheader("Personalized Insights 🧑‍⚕️")
    st.write(response.choices[0].message.content)

def main():
    """Main Streamlit app function with improved UI."""


    uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"], help="Max: 5MB")
    st.title("🏥 Health Recommendation System")
    user_details = enter_details()
    if uploaded_file and user_details:
        image_path = process_uploaded_image(uploaded_file)
        if image_path:
            analyze_image(image_path, user_details)
        else:
            st.error("❌ Image processing failed.")
    else:
        st.warning("Upload an image and enter details to proceed.")




if __name__ == "__main__":
    main()
