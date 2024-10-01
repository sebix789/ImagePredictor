from pymongo import MongoClient
from dotenv import load_doten
import os

mongo_uri = os.getenv("MONGO_URI")

def connect():
    client = MongoClient(mongo_uri)
    db = client["image_predictor"]
    return db

def save_perdiction(image_name, prediction):
    db = connect()
    collection = db["predictions"]
    collection.insert_one({"image_name": image_name, "prediction": prediction})
    
def get_all_predictions():
    db = connect()
    collection = db["predictions"]
    return list(collection.find())