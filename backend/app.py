from flask import Flask, request, jsonify
from flask_cors import CORS
from model.model import load_model
from database import save_prediction, get_all_predictions, update_feedback
from utilities.label_formatter import format_breeds
from PIL import Image
import numpy as np
import tensorflow_datasets as tfds
import io

app = Flask(__name__)
CORS(app)
model = load_model()

datasets, info = tfds.load('stanford_dogs', with_info=True)
labels = info.features['label'].names

@app.route('/')
def index():
    return "Welcome to the Image Classification API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:        
        # Open the image file
        img = Image.open(file).convert('RGB').resize((224, 224))
        
        # Convert image to numpy array
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)
        
        # Start prediction
        prediction = model.predict(img_array)
        predict_class = np.argmax(prediction, axis=1)[0]
        predicted_breed = labels[predict_class]
        formatted_breed = format_breeds(predicted_breed)
        
        ### Save Image to DB ###
        img_io = io.BytesIO()
    
        if file.mimetype == 'image/jpeg' or file.mimetype == 'image/jpg':
            image_format = 'JPEG'
        elif file.mimetype == 'image/png':
            image_format = 'PNG'
        else:
            return jsonify({"error": "Invalid image format"}), 400

        img.save(img_io, format=image_format)
        img_io.seek(0)
        
        # Save prediction to database
        save_prediction(image_name=file.filename, prediction=formatted_breed, image_data=img_io.read())
        
        return jsonify({'image_name': file.filename, 'prediction': formatted_breed})


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    image_name = data.get('image_name')
    is_correct = data.get('is_correct')
    true_label = data.get('true_label')
    
    update_feedback(image_name, is_correct, true_label)
    
    return jsonify({'status': 'Feedback received'})


@app.route('/labels', methods=['GET'])
def get_label():
    try:
        formatted_labels = [format_breeds(label) for label in labels]
        return jsonify({'labels': labels, 'formatted_labels': formatted_labels})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
 
@app.route('/history', methods=['GET'])
def history():
    try:
        predictions = get_all_predictions()
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
    