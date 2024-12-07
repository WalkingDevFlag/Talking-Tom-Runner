import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

# Updated key mappings
w = [1, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0]
a = [0, 0, 1, 0, 0]
d = [0, 0, 0, 1, 0]
n = [0, 0, 0, 0, 1]

def initialize_training_file():
    """
    Initialize the training data file by checking if existing files are present.
    If a file exists, load its data; otherwise, create a new empty list.
    Returns the starting file name, starting value, and loaded or new training data.
    """
    starting_value = 1

    while True:
        file_name = 'data/training_data-{}.npy'.format(starting_value)

        if os.path.isfile(file_name):
            print('File exists, moving along', starting_value)
            starting_value += 1
        else:
            print('File does not exist, starting fresh!', starting_value)
            break

    if os.path.isfile(file_name):
        print(f"File '{file_name}' exists, loading previous data.")
        training_data = list(np.load(file_name, allow_pickle=True))  # Load as a list for appending
    else:
        print(f"File '{file_name}' does not exist, starting fresh!")
        training_data = []

    return file_name, starting_value, training_data

def keys_to_output(keys):
    '''
    Convert keys to a multi-hot array
     0  1  2  3  4
    [W, S, A, D, NOKEY] boolean values.
    '''
    output = [0, 0, 0, 0, 0]

    if 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = n
    return output

def main():
    file_name, starting_value, training_data = initialize_training_file()

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    print('STARTING!!!')
    while True:
        if not paused:
            screen = grab_screen(region=(0, 40, 1920, 1120))
            # Resize and preprocess screen
            screen = cv2.resize(screen, (480, 270))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen, output])

            if len(training_data) % 1000 == 0:
                print(len(training_data))

                if len(training_data) == 10000:
                    np.save(file_name, np.array(training_data, dtype=object))
                    print('SAVED')
                    training_data = []
                    starting_value += 1
                    file_name = 'data/training_data-{}.npy'.format(starting_value)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

main()





# LEGEND
'''
from directkeys import PressKey,ReleaseKey, W, A, S, D
from draw_lanes import draw_lanes


def roi(img, vertices):
    mask = np.zeros_like(img)   
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(image):
    original_image = image
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to Gray
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300) # Edge Detection
    
    processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
    
    # vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]], np.int32)
    vertices = np.array([[0,1600],[0,700],[225,350],[475,350],[900,700],[900,1600]], np.int32) # [0,1300],[0,700],[225,350],[475,350],[700,700],[700,1300]

    processed_img = roi(processed_img, [vertices])

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                                     rho   theta   thresh  min length, max gap:        
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180,      20,       15)
    m1 = 0
    m2 = 0
    try:
        l1, l2, m1,m2 = draw_lanes(original_image,lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                
                
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return processed_img,original_image, m1, m2

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    last_time = time.time()

    while True:
        screen = grab_screen(region=(0,40,800,1600))
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen,original_image, m1, m2 = process_img(screen)
        #cv2.imshow('window', new_screen)
        cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))


        if m1 < 0 and m2 < 0:
            right()
        elif m1 > 0  and m2 > 0:
            left()
        else:
            straight()

        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
'''