from flask import Flask, request, jsonify
from flask_cors import CORS
from model.model import load_model
from database import save_prediction, get_all_predictions
from utilities.label_formatter import format_breeds
from PIL import Image
import numpy as np
import tensorflow_datasets as tfds

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
        # original_width, original_height = img.size
        # img = img.resize((32, 32))
        
        # Convert image to numpy array
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)
        
        # Start prediction
        prediction = model.predict(img_array)
        predict_class = np.argmax(prediction, axis=1)[0]
        predicted_breed = labels[predict_class]
        formatted_breed = format_breeds(predicted_breed)
        
        ### Detect image format and convert to base64 string ###
        # mimetype = file.mimetype
        # if mimetype == 'image/jpeg' or mimetype == 'image/jpg':
        #     image_format = 'JPEG'
        # elif mimetype == 'image/png':
        #     image_format = 'PNG'
        # else:
        #     return jsonify({"error": "Unsupported image format"}), 400
        
        # buffered = io.BytesIO()
        # img.save(buffered, format=image_format)
        # img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Save prediction to database
        save_prediction(file.filename, formatted_breed)
        
        return jsonify({'image_name': file.filename, 'prediction': formatted_breed})
    
 
@app.route('/history', methods=['GET'])
def history():
    try:
        predictions = get_all_predictions()
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
    