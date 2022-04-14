from cv2 import VideoCapture
import pandas as pd
import cv2
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl

#setting context for fetching the data
if( not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X,y = fetch_openml('mnist_784', version = 1, return_X_y= True)
classes = ['0', '1', '2', '3', '4', '5', '6' ,'7', '8', '9']
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 2500, train_size = 7500, random_state = 9)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_pred, y_test)
print("the accuracy is : ", accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #drawing a box in the centre of the video
        height,width = gray.shape()
        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upper_left, bottom_right, (0,255,0), 2)
        roi = gray[upper_left[1]: bottom_right[1], upper_left[0]:bottom_right[0]]
        im_pill = Image.fromarray(roi)

        # convert to grayscale image - 'L' format means each pixel is 
        # represented by a single value from 0 to 255
        image_bw = im_pill.convert('L')
        image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
        pixel_filter = 20
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("Predicted class is: ", test_pred)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF== ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cap.destroyAllWindows()




        

        

        
