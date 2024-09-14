"""
Flask application for house price prediction using a simple neural network model.
"""

import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model from a pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Extract model parameters from the loaded data
W1 = model['W1']
b1 = model['b1']
W2 = model['W2']
b2 = model['b2']
x_mean = model['X_mean']
x_std = model['X_std']
y_mean = model['y_mean']
y_std = model['y_std']

def relu(z_value):
    """
    ReLU activation function.

    Args:
        z_value (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The result of applying ReLU element-wise to the input.
    """
    return np.maximum(0, z_value)

def forward_propagation(input_x):
    """
    Perform forward propagation through the neural network.

    Args:
        input_x (numpy.ndarray): The input feature matrix.

    Returns:
        numpy.ndarray: The predicted output.
    """
    z1_value = np.dot(input_x, W1) + b1
    a1_value = relu(z1_value)
    z2_value = np.dot(a1_value, W2) + b2
    return z2_value

def denormalize(y_pred, mean_value, std_value):
    """
    Denormalize the predictions based on the original mean and standard deviation.

    Args:
        y_pred (numpy.ndarray): The normalized predicted values.
        mean_value (float): The original mean value of the target variable.
        std_value (float): The original standard deviation of the target variable.

    Returns:
        numpy.ndarray: The denormalized predictions.
    """
    return (y_pred * std_value) + mean_value

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Handle the home page where users can input data for prediction.

    Returns:
        str: Rendered HTML template with the prediction result or form.
    """
    if request.method == 'POST':
        try:
            # Get input values from the form and process them
            area = float(request.form['area'])
            bedrooms = float(request.form['bedrooms'])
            bathrooms = float(request.form['bathrooms'])
            stories = float(request.form['stories'])
            mainroad = 1 if request.form['mainroad'].lower() == 'yes' else 0
            
            # Normalize the input data
            input_x = np.array([[area, bedrooms, bathrooms, stories, mainroad]])
            input_x = (input_x - x_mean) / x_std
            
            # Predict and denormalize the price
            y_pred = forward_propagation(input_x)
            y_pred = denormalize(y_pred, y_mean, y_std)
            
            # Return the prediction result
            return render_template(
                'predict.html', prediction=f"Predicted Price: {y_pred[0][0]:.2f}"
            )
        
        except ValueError as error:
            # Handle errors and display them on the index page
            return render_template('index.html', prediction=f"Error: {str(error)}")
    
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    """
    Render the prediction result page.

    Returns:
        str: Rendered HTML template for the prediction page.
    """
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
