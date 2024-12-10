import streamlit as st
import requests
from PIL import Image
import io

# Set up the Streamlit page
st.set_page_config(page_title="Image Classification App", layout="wide")

# Title of the app
st.title("Image Classification App")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to bytes for sending to the API
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Define the API URL
    api_url = "http://localhost:8001/predict"

    # Create a request to the FastAPI backend
    files = {'file': ('image.png', img_byte_arr, 'image/png')}
    
    with st.spinner('Classifying...'):
        try:
            response = requests.post(api_url, files=files , headers={'Origin': 'http://localhost:8501'})
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            # Debug logging
            st.write("Response status code:", response.status_code)
            st.write("Response content:", response.content)
            
            result = response.json()
            st.write("Parsed result:", result)
            
            if result is not None and 'class' in result:
                st.success(f"Prediction: {result['class']}")
                st.json(result)  # Display the full result as JSON
            else:
                st.error("Unexpected response format from the API")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the server: {str(e)}")
            if hasattr(e, 'response'):
                st.error(f"Response content: {e.response.text}")
        except ValueError as e:
            st.error(f"Error processing server response: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# Instructions for users
st.write("Upload an image to get started!")