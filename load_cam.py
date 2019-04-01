# -*- coding: utf-8 -*-
import cv2
from fnload_test import load_and_test

cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 	
    cv2.imwrite('new.jpg',frame)
    load_and_test('new.jpg')


