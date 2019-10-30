 
import cv2
 
cascade_src = 'D:/C-Drive/opencv3.4.3/build/etc/haarcascades/haarcascade_upperbody.xml'
#cascade_src = 'D:/C-Drive/opencv3.4.3/build/etc/haarcascades/haarcascade_fullbody.xml'
video_src = 'D:/PROJECTS/Python/HUMAN COUNT/dataset/VID_20191029_185101.mp4'
video_src = 'E:/Google Drive/PCDS/106_20150509_back/noisy/uncrowd/2015_05_09_15_05_40BackColor.avi'


cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1,3)
#    x1=0
#    y1=0
#    w1=0
#    h1=0
    for (x,y,w,h) in cars:
#        if((w1*h1)<(w*h)):
#            x1=x
#            y1=y
#            w1=w
#            h1=h
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
#    cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)      
    
    cv2.imshow('img', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()