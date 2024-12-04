import numpy as np
from PIL import ImageGrab
import cv2
import time
import mss
from directkeys import PressKey, ReleaseKey, W, A, S, D

def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) # convert to grayscale
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300) # edge detection
    return processed_img


def screen_record(): 
    last_time = time.time()
    while True:
        screen =  np.array(ImageGrab.grab(bbox=(0,40,700,1600)))
        new_screen = process_img(screen)

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()    
        cv2.imshow('window2', new_screen)
        # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def screen_record_mss():
    last_time = time.time()
    with mss.mss() as sct:
        monitor = {"top": 40, "left": 0, "width": 700, "height": 1600}  # Adjust width/height to match the region
        while True:
            screen = np.array(sct.grab(monitor))
            screen = process_img(screen)

            print(f"Loop took {time.time() - last_time:.4f} seconds")
            last_time = time.time()

            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
            cv2.imshow('window', screen_bgr)
            # Exit the loop when 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

def main():
    #print("Starting screen recording. Press 'q' to stop.")
    #screen_record_mss()

    # gives us time to get situated in the game
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    print('slide')
    PressKey(D) 
    time.sleep(3)
    print('jump')
    PressKey(A) 

if __name__ == "__main__":
    main()