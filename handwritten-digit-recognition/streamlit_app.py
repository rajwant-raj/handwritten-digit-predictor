import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2  # OpenCV for image processing
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------------------------------------
# Page Configuration and Custom CSS
# --------------------------------------------------------------------------------

# Set page config
st.set_page_config(page_title="Digit Recognizer", layout="centered")

# Inject custom CSS for styling
def local_css():
    st.markdown("""
        <style>
            /* Import a Google Font */
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
            
            body {
                font-family: 'Roboto', sans-serif;
            }

            /* Main container styling */
            .main .block-container {
                max-width: 700px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            /* Custom Title */
            .title {
                font-size: 3rem;
                font-weight: 700;
                text-align: center;
                color: #2c3e50;
                margin-bottom: 0.5rem;
            }
            
            /* Custom Subheader/Instructions */
            .instructions {
                font-size: 1.1rem;
                font-weight: 300;
                text-align: center;
                color: #555;
                margin-bottom: 2rem;
            }

            /* Styling the canvas container */
            .stCanvas > div {
                border: 2px solid #eee;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }
            
            /* Prediction Box */
            .prediction-box {
                background-color: #f4f8fa;
                border: 2px solid #e0e6ed;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                margin-top: 2rem;
            }
            
            .prediction-header {
                font-size: 1.5rem;
                font-weight: 400;
                color: #555;
            }
            
            .prediction-digit {
                font-size: 6rem;
                font-weight: 700;
                color: #3498db; /* A nice blue */
                margin: 1rem 0;
            }
            
            .prediction-confidence {
                font-size: 1.1rem;
                font-weight: 300;
                color: #777;
            }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --------------------------------------------------------------------------------
# Model Loading
# --------------------------------------------------------------------------------

# Cache the model loading
@st.cache_resource
def load_keras_model():
    """Load the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model("my_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_keras_model()

# --------------------------------------------------------------------------------
# Image Preprocessing
# --------------------------------------------------------------------------------

def preprocess_image(img_data):
    """
    Preprocess the canvas image to match the MNIST model's input requirements.
    (28x28, grayscale, inverted, normalized)
    """
    # 1. Convert canvas RGBA to Grayscale
    # The canvas output is (height, width, 4) - RGBA
    # We use OpenCV to convert it to a 3-channel BGR image first
    img_rgba = np.array(img_data)
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Resize to 28x28
    # Use INTER_AREA for shrinking, which is generally best
    resized_img = cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 3. Invert colors
    # The canvas is black (0) on white (255)
    # MNIST is white (255) on black (0)
    inverted_img = 255 - resized_img
    
    # 4. Normalize
    normalized_img = inverted_img.astype('float32') / 255.0
    
    # 5. Reshape for the model
    # Model expects (batch_size, height, width, channels)
    reshaped_img = normalized_img.reshape(1, 28, 28, 1)
    
    return reshaped_img, inverted_img # Return processed image for debug

# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------

if model is not None:
    # Custom HTML Title and Instructions
    st.markdown('<p class="title"><h2><b>Handwritten Digit Predictor</b></h2></p>', unsafe_allow_html=True)
    st.markdown('<p class="instructions">Draw a digit from 0 to 9 on the canvas below and click "Predict".</p>', unsafe_allow_html=True)

    # Create two columns: one for the canvas, one for the controls/output
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create a drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",  # Fixed fill color with opacity
            stroke_width=20,  # Make the drawing line thick
            stroke_color="#000000", # Black color
            background_color="#FFFFFF",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col2:
        # Placeholder for controls and results
        predict_button = st.button("Predict", use_container_width=True, type="primary")
        clear_button = st.button("Clear Canvas", use_container_width=True)
        
        # Add a small debug view
        st.write("What the model sees (28x28):")
        debug_image_placeholder = st.empty()

        # Placeholder for the prediction result
        result_placeholder = st.empty()

    # --------------------------------------------------------------------------------
    # Prediction Logic
    # --------------------------------------------------------------------------------

    if clear_button:
        # Note: Clearing the canvas is handled by Streamlit's session state magic
        # by re-rendering. We just need to clear our placeholders.
        result_placeholder.empty()
        debug_image_placeholder.empty()

    if predict_button:
        if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
            # Get the image data from the canvas
            img_data = canvas_result.image_data
            
            # Preprocess the image
            processed_img, debug_img = preprocess_image(img_data)
            
            # Make prediction
            prediction = model.predict(processed_img)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)

            # Display the debug image (what the model sees)
            debug_image_placeholder.image(debug_img, width=150)
            
            # Display the prediction using custom HTML
            result_placeholder.markdown(f"""
                <div class="prediction-box">
                    <p class="prediction-header">Prediction:</p>
                    <p class="prediction-digit">{predicted_digit}</p>
                    <p class="prediction-confidence">Confidence: {confidence*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
        else:
            result_placeholder.warning("Please draw a digit on the canvas first.")
else:
    st.error("Model file 'my_model.keras' not found. Please make sure it's in the same directory.")

st.markdown("---")
st.markdown("<center>Made by rajwant-raj ✨ | CNN + MNIST + Streamlit</center>", unsafe_allow_html=True)