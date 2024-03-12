import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 40
imgSize = 300
labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255gmail

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        aspectRatio = w/h
        if aspectRatio > 1:
            new_w = imgSize
            new_h = int(imgSize/aspectRatio)
            predicition, index = classifier.getPrediction(img)
            print(predicition, index)

        else:
            new_h = imgSize
            new_w = int(imgSize*aspectRatio)
        
        imgCropResized = cv2.resize(imgCrop, (new_w, new_h))

        start_x = (imgSize - new_w) //2
        start_y = (imgSize - new_h) //2

        imgWhite[start_y:start_y+new_h, start_x:start_x+new_w] = imgCropResized


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
        
    cv2.imshow("Image", img)
    cv2.waitKey(1)