from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

W1 = model['W1']
b1 = model['b1']
W2 = model['W2']
b2 = model['b2']
X_mean = model['X_mean']
X_std = model['X_std']
y_mean = model['y_mean']
y_std = model['y_std']

# Activation function (ReLU)
def relu(z):
    return np.maximum(0, z)

# Forward propagation
def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    return Z2

# Denormalize the predictions
def denormalize(y_pred, y_mean, y_std):
    return (y_pred * y_std) + y_mean

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get input values from the form
            area = float(request.form['area'])
            bedrooms = float(request.form['bedrooms'])
            bathrooms = float(request.form['bathrooms'])
            stories = float(request.form['stories'])
            mainroad = 1 if request.form['mainroad'].lower() == 'yes' else 0
            
            # Normalize the input data
            X = np.array([[area, bedrooms, bathrooms, stories, mainroad]])
            X = (X - X_mean) / X_std
            
            # Predict and denormalize
            y_pred = forward_propagation(X)
            y_pred = denormalize(y_pred, y_mean, y_std)
            
            # Redirect to the predict page with the result
            return render_template('predict.html', prediction=f"Predicted Price: {y_pred[0][0]:.2f}")
        
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")
    
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
