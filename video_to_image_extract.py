# -*- coding: utf-8 -*-

import cv2
import os


video_src = 'E:/PROJECT ALL/kaggle/project/human Count/dataset/Project 1.avi'

cap = cv2.VideoCapture(video_src)

outputpath = 'E:/PROJECT ALL/kaggle/project/human Count/dataset/extract/Project 1/'
count=0
while True:
    ret, img = cap.read()
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    cv2.imwrite(os.path.join(outputpath , 'hc-'+str(count)+'.jpg'), img)
#    cv2.imwrite("frame%d.jpg" % count, image)
    count+=1
    
   

cv2.destroyAllWindows()