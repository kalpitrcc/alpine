import requests
import pytest

# Define the API base URL
BASE_URL = "http://your-api-url.com"  --> replace with env variable from gateway

# Positive Test Case 1: Testing "/" endpoint
def test_hello_world_endpoint():
    response = requests.get(BASE_URL + "/")
    assert response.status_code == 200
    assert response.text == "hello world"

# Positive Test Case 2: Testing "/predict" endpoint with a valid image
def test_predict_valid_image():
    image_data = open("valid_image.jpg", "rb").read()
    response = requests.post(BASE_URL + "/predict", data=image_data)
    assert response.status_code == 200

# Negative Test Case 1: Testing "/predict" endpoint with no image
def test_predict_no_image():
    response = requests.post(BASE_URL + "/predict")
    assert response.status_code == 400  # You can choose the appropriate status code
    # Add assertions to validate the error message or response content

# Negative Test Case 2: Testing an invalid endpoint
def test_invalid_endpoint():
    response = requests.get(BASE_URL + "/invalid_endpoint")
    assert response.status_code == 404 

if name == "__main__":
    pytest.main()
