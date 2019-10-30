# -*- coding: utf-8 -*-

import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from PIL import ImageFont, ImageDraw 
import numpy as np
  
 
 
video_src = 'D:/PROJECTS/Python/HUMAN COUNT/dataset/VID_20191029_185101.mp4'
video_src = 'E:/Google Drive/PCDS/106_20150509_back/noisy/uncrowd/2015_05_09_15_05_40BackColor.avi'
outputpath = 'D:/PROJECTS/Python/dataExtract/VID7/'
 

 
 

cap = cv2.VideoCapture(video_src)
 
count=0
ret_pre, img_pre = cap.read()
img_pre_gray = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
th2_pre = cv2.adaptiveThreshold(img_pre_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    img_sub=img_pre_gray-img_gray
    
    ret,thresh = cv2.threshold(img_sub,50,120,cv2.THRESH_TRUNC)
    th2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    
    img_sub=img_pre-img
    
    th2_sub=th2_pre-th2
    ret,thresh11 = cv2.threshold(th2_sub,127,255,cv2.THRESH_TRUNC)
     
    cv2.imshow('img', th2_sub)
#    cv2.imshow('img',cv2.cvtColor(img,cv2.COLOR_BAYER_GR2RGB ) )
    img_pre_gray=img_gray
    th2_pre=th2
    img_pre=img
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()