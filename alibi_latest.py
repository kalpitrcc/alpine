##############
import os
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import logging
import io
from alibi_detect.cd import TabularDrift
from alibi_detect.utils.saving import save_detector, load_detector


# Add logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = f"{BASE_DIR}/model"
DRIFT_DIR = f"{BASE_DIR}/drift"
IMAGE_FOLDER = f"{BASE_DIR}/sample_images"

def load_model():
    model = tf.keras.models.load_model(MODEL_DIR)
    return model

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def preprocess_images(file_paths):
    images = []
    for file_path in file_paths:
        # Load and preprocess the image based on your specific preprocessing steps
        img = load_and_preprocess_image(file_path)
        images.append(img)
    return np.array(images)

def load_and_preprocess_image(file_path):

    image = Image.open(file_path)    
    image = image.convert('L').resize((28, 28))
    image_array = np.array(image)
    image_array = image_array.reshape(1, 28, 28) / 255.0
    
    return image_array

# Function to read all images from a folder
def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_paths = [os.path.join(folder_path, f) for f in image_files]
    return image_paths

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):

    # Load previously saved drift detector or create a new one
    if not os.path.exists(DRIFT_DIR):
        print(f"The drift folder '{DRIFT_DIR}' doesn't exist. Creating a new detector.")
        os.mkdir(DRIFT_DIR) 
        # Initialize drift detector with initial data
        image_paths = read_images_from_folder(IMAGE_FOLDER)
        X_all = preprocess_images(image_paths)
        # Build drift detector
        cd = TabularDrift(X_all, p_val=.05)
        # Save the detector for future use
        save_detector(cd, DRIFT_DIR)
    else:
        print(f"The drift folder '{DRIFT_DIR}' already exists. Loading the existing detector.")   
        cd = load_detector(DRIFT_DIR)
        
    try:
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents)).convert('L').resize((28, 28))
        image_array = np.array(image)
        image_array = image_array.reshape(1, 28, 28) / 255.0

        model = load_model()

        predictions = model.predict(image_array)
        predicted_label = np.argmax(predictions[0])

        # Detect drift
        drifts = cd.predict(image_array)

        # Print the results
        print('Drift detection results:')
        print(drifts)

        return {"predicted_label": int(predicted_label), "class_name": class_names[predicted_label]}
    except Exception as e:
        logger.error(f"Error processing the image: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image")

if _name_ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
