from database import connect
from gridfs import GridFS
import os
import json
import zipfile
import shutil




def export_data(directory='temp', file_name='data.zip'):
    db = connect()
    collection = db["predictions"]
    fs = GridFS(db)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    predictions = list(collection.find())
    metadata = []
    
    for pred in predictions:
        image_id = pred["image_id"]
        image_name = pred["image_name"]
        prediction = pred["prediction"]
        is_correct = pred["is_correct"]
        true_label = pred.get("true_label", None)
        
        image_data = fs.get(image_id).read()
        
        image_path = os.path.join(directory, image_name)
        with open(image_path, 'wb') as f:
            f.write(image_data)
            
        
        metadata.append({
            "image_name": image_name,
            "prediction": prediction,
            "is_correct": is_correct,
            "true_label": true_label
        })
        
    metadata_path = os.path.join(directory, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f,indent=4)
    
    with zipfile.ZipFile(file_name, 'w') as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory))
    
    print(f"Data exported to file {file_name}")
    
    shutil.rmtree(directory)
    print("Temp folder removed")
    
export_data()