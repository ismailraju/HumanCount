# -*- coding: utf-8 -*-

import cv2
import os




#video_src = 'E:/PROJECT ALL/kaggle/project/human Count/dataset/Project 1.avi'
video_src = 'E:/Google Drive/PCDS/106_20150509_back/normal/crowd/'

for file in os.listdir(video_src):
    if file.endswith(".avi"):
        print(file)
        
        
        cap = cv2.VideoCapture(video_src+file)
        
        #outputpath = 'E:/PROJECT ALL/kaggle/project/human Count/dataset/extract/Project 1/'
        outputpath = 'E:/Google Drive/PCDS/106_20150509_back/normal/crowd/'+file.split(".avi")[0]
        count=0
        while True:
            ret, img = cap.read()
            ret, img = cap.read()
            if (type(img) == type(None)):
                break
            
            
            file_output_path = os.path.join(outputpath+"/" ,"hc-"+str(count)+'.jpg')
            directory = os.path.dirname(file_output_path)
            
            try:
                os.stat(directory)
            except:
                os.mkdir(directory)
            
            cv2.imwrite(file_output_path, img)
        #    cv2.imwrite("frame%d.jpg" % count, image)
            count+=1
            
           

cv2.destroyAllWindows()