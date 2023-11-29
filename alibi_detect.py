import numpy as np
from tensorflow.keras.models import load_model
from alibi_detect.cd import TabularDrift
from alibi_detect.utils.saving import save_detector, load_detector

# Load your MNIST model (you may need to adjust this based on your actual model loading code)
model = load_model('path/to/your/mnist_model.h5')

# Function to preprocess image files
def preprocess_images(file_paths):
    images = []
    for file_path in file_paths:
        # Load and preprocess the image based on your specific preprocessing steps
        img = load_and_preprocess_image(file_path)
        images.append(img)
    return np.array(images)

# Function to load and preprocess a single image
def load_and_preprocess_image(file_path):
    # Implement your image loading and preprocessing logic
    # Make sure the image is resized and normalized in the same way as during model training
    return preprocessed_image

# Load previously saved drift detector or create a new one
try:
    cd = load_detector('path/to/your/detector')
except FileNotFoundError:
    # Initialize drift detector with initial data
    X_initial = preprocess_images(['path/to/initial_data_image.jpg'])
    cd = TabularDrift(X_initial, p_val=.05, backend='tensorflow', model=model)
    save_detector(cd, 'path/to/your/detector')

# Live input file paths
live_files = ['path/to/live_image_1.jpg', 'path/to/live_image_2.jpg', ...]

# Preprocess live input images
X_live = preprocess_images(live_files)

# Detect drift
drifts = cd.predict(X_live)

# Print the results
print('Drift detection results:')
print(drifts)
