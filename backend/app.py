from flask import Flask, request, jsonify
from model import load_model
from database import save_perdiction, get_all_predictions
from PIL import Image
import numpy as np
import os


app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    # get files from client
    file = file.request['file']
    img = Image.open(file).resize((32, 32))
    
    # convert image to numpy array
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)
    
    
    # start prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    
    # save prediction to database
    save_perdiction(file.filename, int(predicted_class))
    
 
@app.route('/history', methods=['GET'])
def history():
    predictions = get_all_predictions()
    return jsonify({'prediction': predictions})


if __name__ == '__main__':
    app.run(debug=True)
    
    print('Server is runnig on port: 5000')
    