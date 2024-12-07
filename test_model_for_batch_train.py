import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

def cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Specify model path
MODEL_PATH = 'models/talking_tom_cnn_model.keras'
DATA_FOLDER = 'data/training_batches'  # Folder containing training data
MODEL_SAVE_INTERVAL = 10  # Save model every 10 epochs
BATCH_SIZE = 32
EPOCHS = 100

# Load pre-trained model or create a new one
input_shape = (80, 60, 1)
if os.path.exists(MODEL_PATH):
    print(f"Loading pre-trained model from {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a new model.")
        model = cnn_model(input_shape)
else:
    print("No pre-trained model found. Creating a new model.")
    model = cnn_model(input_shape)

# Initialize ModelCheckpoint to save model periodically
checkpoint_callback = ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

# Iterate through training data files
data_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.npy')]
if not data_files:
    raise FileNotFoundError("No training data files found in the specified folder.")

for file_idx, file_name in enumerate(data_files):
    try:
        print(f"Loading data from file: {file_name}")
        file_path = os.path.join(DATA_FOLDER, file_name)
        train_data = np.load(file_path, allow_pickle=True)

        # Prepare data
        X = np.array([i[0] for i in train_data]).reshape(-1, 80, 60, 1)
        Y = np.array([i[1] for i in train_data])  # Labels (one-hot encoded)

        # Split into training and testing sets
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        print(f"Training on data file {file_idx + 1}/{len(data_files)}: {file_name}")

        # Train model
        model.fit(
            X_train, Y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, Y_val),
            callbacks=[checkpoint_callback]
        )

    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")
        continue

# Save final model state
try:
    print(f"Saving final model to {MODEL_PATH}")
    model.save(MODEL_PATH)
except Exception as e:
    print(f"Error saving final model: {e}")

print("Training complete.")
