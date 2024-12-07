import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from getkeys import key_check
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

# Define the CNN model
def cnn_model(input_shape):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output for the dense layers
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output layer (5 classes: W, S, A, D, N)
    model.add(layers.Dense(5, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Define the model file name
MODEL_NAME = 'models/talking_tom_cnn_model_1.h5'

# Load the trained model
model = load_model(MODEL_NAME)

def jump():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)

def right():
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)

def slide():
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def nokey():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while True:
        if not paused:
            # 800x1600 windowed mode
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 1600)))
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()

            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80, 60))

            # Reshape input for prediction
            screen = screen.reshape(1, 80, 60, 1)

            moves = list(np.around(model.predict(screen)[0]))
            # output = [W, A, S, D, N]
            if moves == [1, 0, 0, 0, 0]:
                jump()
            elif moves == [0, 1, 0, 0, 0]:
                left()
            elif moves == [0, 0, 1, 0, 0]:
                slide()
            elif moves == [0, 0, 0, 1, 0]:
                right()
            elif moves == [0, 0, 0, 0, 1]:
                nokey()

        keys = key_check()

        # Pause and unpause using 'T'
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                nokey()
                time.sleep(1)

if __name__ == "__main__":
    main()