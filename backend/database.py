from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

def connect():
    client = MongoClient(mongo_uri)
    db = client["image_predictor"]
    return db

def save_prediction(image_name, prediction):
    db = connect()
    collection = db["predictions"]
    collection.insert_one({
        "image_name": image_name,
        "prediction": prediction,
        # "image_data": image_data,
        # "original_width": original_width,
        # "original_height": original_height
    })
    
def get_all_predictions():
    db = connect()
    collection = db["predictions"]
    predictions = list(collection.find())
    for prediction in predictions:
        prediction["_id"] = str(prediction["_id"])
    return predictions