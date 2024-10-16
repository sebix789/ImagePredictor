from flask import Flask, request, jsonify
from flask_cors import CORS
from model.model import load_model
from database import save_prediction, get_all_predictions, update_feedback
from utilities.label_formatter import format_breeds
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import tensorflow_datasets as tfds
import io
import os
import re


load_dotenv()
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
        save_prediction(image_name=file.filename, prediction=predicted_breed, image_data=img_io.read())
        
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
    

@app.route('/auto-feedback', methods=['POST'])
def auto_feedback():
    results = []
    correct_predictions = 0
    total_predictions = 0    
    
    upload_path = os.getenv('UPLOAD_PATH')
    for filename in os.listdir(upload_path):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(upload_path, filename)
            
            img = Image.open(file_path).convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 224, 224, 3)

            prediction = model.predict(img_array)
            predict_class = np.argmax(prediction, axis=1)[0]
            predicted_breed = labels[predict_class]

            original_label_match = re.match(r"([a-zA-Z_]+)", filename)
            if original_label_match:
                original_label = original_label_match.group(1).lower()
            else:
                original_label = filename.split(".")[0].lower()

            formatted_original_label = original_label.replace('_', ' ').lower()
            predicted_breed_name = predicted_breed.split("-")[1].replace('_', ' ').lower()
            
            print(f"Original Label: {formatted_original_label}")
            print(f"Predicted Breed: {predicted_breed_name}")
            
            is_correct = (predicted_breed_name.lower() == formatted_original_label.lower())
            total_predictions += 1
            
            if is_correct:
                correct_predictions += 1

            if not is_correct:
                matching_label = None
                for label in labels:
                    formatted_label = label.split("-")[1].replace('_', ' ').lower()
                    if formatted_label == formatted_original_label:
                        matching_label = label
                        break

                if matching_label:
                    true_label = matching_label
                else:
                    true_label = None
            else:
                true_label = None

            img_io = io.BytesIO()
            img.save(img_io, format='JPEG')
            img_io.seek(0)

            save_prediction(image_name=filename, prediction=predicted_breed, image_data=img_io.read(), is_correct=is_correct, true_label=true_label)
            
            results.append({
                'image_name': filename,
                'predicted_breed': predicted_breed,
                'is_correct': is_correct,
                'true_label': true_label
            })
            
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
    else:
        accuracy = 0.0
        
    summary = {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy
    }
    
    print(f"Total Predictions: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Accuracy: {accuracy}")
    
    return jsonify({ 'results': results, 'summary': summary })


if __name__ == '__main__':
    app.run(debug=True)
    