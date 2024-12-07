import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from getkeys import key_check
from tensorflow.keras.models import load_model

# Load the pre-trained model
# Define the model file name
MODEL_NAME = 'models/talking_tom_cnn_model.keras'

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
            print(moves)

            
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