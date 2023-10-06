import os
import requests
import pytest

# Get the API base URL from an environment variable
BASE_URL = os.getenv("API_BASE_URL", "http://your-api-url.com")

# Positive Test Case 1: Testing "/" endpoint
def test_hello_world_endpoint():
    response = requests.get(BASE_URL + "/")
    assert response.status_code == 200
    assert response.text == "hello world"

# Positive Test Case 2: Testing "/predict" endpoint with a valid image
def test_predict_valid_image():
    with open("valid_image.jpg", "rb") as image_file:
        files = {"image": image_file}
        response = requests.post(BASE_URL + "/predict", files=files)
    assert response.status_code == 200

# Negative Test Case 1: Testing "/predict" endpoint with no image
def test_predict_no_image():
    response = requests.post(BASE_URL + "/predict")
    assert response.status_code == 400
    # Add assertions to validate the error message or response content if applicable

# Negative Test Case 2: Testing an invalid endpoint
def test_invalid_endpoint():
    response = requests.get(BASE_URL + "/invalid_endpoint")
    assert response.status_code == 404 

if _name_ == "__main__":
    pytest.main()
