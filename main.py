import os
import requests
import pytest
import logging

# Configure logging
log = logging.getLogger(__name__)

# Define the API base URL
BASE_URL = os.environ.get("GATEWAY_URL")

# Positive Test Case 1: Testing "/" endpoint
def test_hello_world_endpoint():
    response = requests.get(BASE_URL + "/", verify=False)
    assert response.status_code == 200
    assert response.text == "Hello, World"  # Fix the assertion message
    log.info(f"Response status code: {response.status_code}")

# Positive Test Case 2: Testing "/predict" endpoint with a valid image
def test_predict_valid_image():
    dir_list = os.listdir("/tests/sample_images/")
    print(dir_list)
    with open("/tests/sample_images/sample-0.png", "rb") as image_file:
        image_data = image_file.read()
        response = requests.post(BASE_URL + "/predict", data=image_data, verify=False)
    assert response.status_code == 200
    log.info(f"Response status code: {response.status_code}")
    print(response)

# Negative Test Case 1: Testing "/predict" endpoint with no image
def test_predict_no_image():
    response = requests.post(BASE_URL + "/predict", verify=False)
    assert response.status_code == 400
    log.info(f"Response status code: {response.status_code}")
    print(response)

# Negative Test Case 2: Testing an invalid endpoint
def test_invalid_endpoint():
    response = requests.get(BASE_URL + "/invalid_endpoint", verify=False)
    assert response.status_code == 404
    log.info(f"Response status code: {response.status_code}")

if name == "__main__":
    pytest.main()
