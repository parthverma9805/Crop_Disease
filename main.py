import os
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Constants
IMG_SIZE = 128
# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'crop_disease_model.h5')
CLASSES_PATH = os.path.join(BASE_DIR, 'classes.npy')

# Load the trained model and classes
model = None
classes = None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        model = load_model(MODEL_PATH)
        classes = np.load(CLASSES_PATH, allow_pickle=True)
        print("Model and classes loaded successfully.")
    else:
        print("Model or classes files not found. Please ensure 'crop_disease_model.h5' and 'classes.npy' are in the same directory as this script.")
except Exception as e:
    print(f"Error loading model or classes: {e}")

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('a.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives base64 image data from the front-end, processes it,
    and returns a prediction.
    """
    if 'image' not in request.form:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Check if the model and classes were loaded successfully at startup
    if model is None or classes is None:
        return jsonify({
            'error': 'Server error: Model or classes files are missing. Please upload the necessary files.'
        }), 500

    image_data_b64 = request.form['image']

    try:
        # Decode the base64 string
        # The data starts with "data:image/png;base64,...", so we split to get the raw data
        encoded_data = image_data_b64.split(',')[1]
        decoded_data = base64.b64decode(encoded_data)
        
        # Open the image using Pillow and resize
        image = Image.open(BytesIO(decoded_data)).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to a numpy array and normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make a prediction
        prediction_probabilities = model.predict(img_array)
        predicted_class_index = np.argmax(prediction_probabilities)
        
        # Extract the predicted class name and confidence
        # The class names might have underscores; we clean them up for display.
        predicted_class_name = str(classes[predicted_class_index]).replace('__', ' ').replace('_', ' ')
        confidence = float(prediction_probabilities[0][predicted_class_index]) * 100

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': f'{confidence:.2f}%'
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
