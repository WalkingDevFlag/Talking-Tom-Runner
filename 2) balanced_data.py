import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('data/training_data-7.npy', allow_pickle=True)


df = pd.DataFrame(train_data)
print(df.shape)

print(df.head())
print(Counter(df[1].apply(str)))

'''
w = [1,0,0,0,0]
s = [0,1,0,0,0]
a = [0,0,1,0,0]
d = [0,0,0,1,0]
n = [0,0,0,0,1]

[W, A, S, D, nokey(n)]
'''
lefts = []
rights = []
jumps = []
slides = []
nokeys = []

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0,0,0]:
        jumps.append([img, choice])
    elif choice == [0,1,0,0,0]:
        lefts.append([img, choice])
    elif choice == [0,0,1,0,0]:
        slides.append([img, choice])
    elif choice == [0,0,0,1,0]:
        rights.append([img, choice])
    else:
        nokeys.append([img, choice])

jumps = jumps[:len(lefts)][:len(rights)][:len(slides)]
lefts = lefts[:len(jumps)]
rights = rights[:len(jumps)]
slides = slides[:len(jumps)]

final_data = jumps + lefts + rights + slides
shuffle(final_data)

np.save('data/training_data_v7.3.npy', np.array(final_data, dtype=object))
    

'''
for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test',img)
    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
'''