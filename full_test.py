import pytest
import numpy as np
import pandas as pd
import pickle
from app import app, forward_propagation, denormalize

# Load the model data for testing
@pytest.fixture(scope='module')
def model_data():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

def test_forward_propagation(model_data):
    # Example input (standardized)
    X = np.array([[1, 2, 3, 4, 1]])  # Should match the number of features in your input data (5)

    # Standardize the input using the model's mean and std
    X = (X - model_data['X_mean']) / model_data['X_std']

    # Perform forward propagation
    Z2 = forward_propagation(X)  # Assuming forward_propagation only returns Z2

    # Check the shape of the output
    assert Z2.shape == (1, 1), f"Expected Z2 shape (1, 1), but got {Z2.shape}"

    print("Test passed for forward propagation!")

def test_denormalize(model_data):
    y_pred = np.array([[0.5]])
    y_mean = model_data['y_mean']
    y_std = model_data['y_std']
    result = denormalize(y_pred, y_mean, y_std)
    assert result.shape == (1, 1), "Denormalize output shape mismatch."
    assert isinstance(result[0][0], float), "Denormalize result type mismatch."

# Tests for Flask app
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_post(client):
    response = client.post('/', data={
        'area': '1200',
        'bedrooms': '3',
        'bathrooms': '2',
        'stories': '1',
        'mainroad': 'yes'
    })
    print(response.data)  # Print the response data for debugging
    assert b"Predicted Price:" in response.data, "Prediction page not rendered correctly."

def test_home_get(client):
    response = client.get('/')
    assert b"House Price Prediction" in response.data, "Home page title not rendered correctly."
    assert b"Predict" in response.data, "Submit button not rendered correctly."  # Update this line
    assert b"Area:" in response.data, "Area label not rendered correctly."
    assert b"Bedrooms:" in response.data, "Bedrooms label not rendered correctly."
    assert b"Bathrooms:" in response.data, "Bathrooms label not rendered correctly."
    assert b"Stories:" in response.data, "Stories label not rendered correctly."
    assert b"Main Road (yes/no):" in response.data, "Main Road label not rendered correctly."

def test_predict(client):
    response = client.get('/predict')
    assert b"Prediction Result" in response.data, "Predict page title not rendered correctly."
    assert b"Go Back" in response.data, "Go Back button not rendered correctly."
