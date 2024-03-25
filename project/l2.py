import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = cv2.hand_detection.HandDetector()

offset = 20
img_size = 300
counter = 0

folder = "MINOR PROJECT\Data\hello"

while True:
    success, img = cap.read()
    hands = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgCropShape = imgCrop.shape

        aspect_ratio = h/w
        if aspect_ratio > 1:
            k = img_size/h
            w_cal = math.ceil(k*w)

            imgResize = cv2.resize(imgCrop, (w_cal, img_size))
            imgResizeShape = imgResize.shape
            w_gap = math.ceil((img_size-w_cal)/2)

            imgWhite = np.ones((img_size, img_size, 3), np.uint8)*255
            imgWhite[:, w_gap:w_cal+w_gap] = imgResize

        else:
            k = img_size/w
            h_cal = math.ceil(k*h)

            imgResize = cv2.resize(imgCrop, (img_size, h_cal))
            imgResizeShape = imgResize.shape
            h_gap = math.ceil((img_size-h_cal)/2)

            imgWhite = np.ones((img_size, img_size, 3), np.uint8)*255
            imgWhite[h_gap:h_cal+h_gap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)cv2.imshow('ImageWhite', imgWhite)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(counter)

cap.release()
cv2.destroyAllWindows()