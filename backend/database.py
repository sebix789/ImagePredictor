from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import os

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

def connect():
    client = MongoClient(mongo_uri)
    db = client["image_predictor"]
    return db

def save_prediction(image_name, prediction, image_data, is_correct=None, true_label=None):
    db = connect()
    fs = GridFS(db)
    image_id = fs.put(image_data, filename=image_name)
    
    collection = db["predictions"]
    collection.insert_one({
        "image_name": image_name,
        "prediction": prediction,
        "image_id": image_id,
        "is_correct": is_correct,
        "true_label": true_label
    })
    
def get_all_predictions():
    db = connect()
    collection = db["predictions"]
    predictions = list(collection.find())
    for prediction in predictions:
        prediction["_id"] = str(prediction["_id"])
    return predictions

def update_feedback(image_name, is_correct, true_label=None):
    db = connect()
    collection = db["predictions"]
    
    collection.update_one(
        {"image_name": image_name},
        {"$set": {"is_correct": is_correct, "true_label": true_label}}
    )