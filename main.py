import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split  # Can skip sklearn if not using it

# Load and preprocess data
data = pd.read_csv('housing.csv')

x = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad']].values
y = data['price'].values

# Convert 'mainroad' to binary
x[:, 4] = np.where(x[:, 4] == 'yes', 1, 0)

# Normalize features
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x = (x - x_mean) / x_std

# Reshape target variable
y = y.reshape(-1, 1)

# Split the data (optional if not using sklearn)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build a simple neural network using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2)

# Evaluate the model
loss = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
predictions = model.predict(x_test)
print(predictions)
