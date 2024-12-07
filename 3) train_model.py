# train.py
import numpy as np
from sklearn.model_selection import train_test_split
from model import cnn_model_base

# Load the dataset
train_data = np.load('data/training_data_v7.3.npy', allow_pickle=True)

# Prepare the data
X = np.array([i[0] for i in train_data]).reshape(-1, 80, 60, 1)  # 80x60 grayscale images
Y = np.array([i[1] for i in train_data])  # Labels (one-hot encoded)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the model
input_shape = (80, 60, 1)  # 80x60 grayscale images
model = cnn_model_base(input_shape)

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))

# Save the model
model.save('talking_tom_cnn_model.h5')