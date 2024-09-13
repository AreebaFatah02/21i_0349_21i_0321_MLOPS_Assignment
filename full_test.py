import pytest
import numpy as np
import pandas as pd
import pickle
import os
from app import app, forward_propagation, denormalize

# Load the model data for testing
@pytest.fixture(scope='module')
def model_data():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

# Tests for main.py functions
def test_forward_propagation(model_data):
    X = np.array([[1, 2, 3, 4, 1]])  # Example input
    W1 = model_data['W1']
    b1 = model_data['b1']
    W2 = model_data['W2']
    b2 = model_data['b2']
    
    Z1, A1, Z2 = forward_propagation(X, W1, b1, W2, b2)
    assert Z2.shape == (1, 1), "Forward propagation output shape mismatch."

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

def test_home_post(client, model_data):
    response = client.post('/', data={
        'area': 1200,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 1,
        'mainroad': 'yes'
    })
    assert b"Predicted Price:" in response.data, "Prediction page not rendered correctly."

def test_home_get(client):
    response = client.get('/')
    assert b"Submit" in response.data, "Home page not rendered correctly."

def test_predict(client):
    response = client.get('/predict')
    assert b"Prediction" in response.data, "Predict page not rendered correctly."
