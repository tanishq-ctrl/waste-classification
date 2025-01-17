import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model - using the correct path you provided
MODEL_PATH = '/Users/tanishq/Desktop/waste_model/waste_classification_model.h5'
print(f"Attempting to load model from: {MODEL_PATH}")

# Check if file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Please check if the file exists at this location.")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Categories mapping
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image_path):
    """Preprocess the image to match model requirements."""
    # Load and resize image
    img = Image.open(image_path)
    img = img.resize((128, 128))
    
    # Convert to array and preprocess
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize
    
    return img_array

def predict_waste_category(image_path):
    """Predict the waste category for a given image."""
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    
    # Return predicted category
    return CATEGORIES[predicted_class] 