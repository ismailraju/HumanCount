 
import cv2
 

cascade_src = 'E:/PROJECT ALL/kaggle/project/human Count/HumanCountraju/haarcascade/classifier/cascade.xml'

#cascade_src = 'D:/C-Drive/opencv3.4.3/build/etc/haarcascades/haarcascade_fullbody.xml'
video_src = 'E:/PROJECT ALL/kaggle/project/human Count/dataset/Project 1.avi'
video_src ='E:/PROJECT ALL/kaggle/project/human Count/dataset/VID_20191030_180929.mp4'
video_src ='E:/PROJECT ALL/kaggle/project/human Count/dataset/PCDS/106_20150509_back/noisy/crowd/2015_05_09_15_30_14BackColor.avi'
#video_src = 'E:/Google Drive/PCDS/106_20150509_back/noisy/uncrowd/2015_05_09_15_05_40BackColor.avi'


#cascade_src = 'D:/PROJECTS/Python/HUMAN COUNT/human_count_raju/haarcascade/classifier/cascade.xml'
#
#video_src = 'D:/PROJECTS/Python/HUMAN COUNT/dataset/VID_20191030_180929.mp4'

ALLOWED_RECOGNITION_MISSING_FRAME=5



#moveDiff=0

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
 
def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  d = abs(a[4]-b[4])
  return (x, y, w, h,d)


def crossMiddle(x1,x2):
    if(x1<=100 and x2>100):
        return 1
    
    return 0

def drawLines(pre_rects,image):
    moveDiff=0
    cross=0;
    for i in range(1,len(pre_rects)-1):
        moveDiff=moveDiff+(pre_rects[i][0]-pre_rects[i-1][0])
        cross=cross+crossMiddle(pre_rects[i-1][0],pre_rects[i][0])
        lineThickness = 2
        print((pre_rects[i-1][0] , pre_rects[i-1][1]), (pre_rects[i][0], pre_rects[i][1]))
    #    cv2.line(image, (x1, y1), (x2, y2), (0,255,0), lineThickness)
        cv2.line(image, (int(pre_rects[i-1][0] +(pre_rects[i-1][2] /2)), int(pre_rects[i-1][1] +(pre_rects[i-1][3] /2))), (int(pre_rects[i][0] +(pre_rects[i][2] /2)), int(pre_rects[i][1] +(pre_rects[i][3] /2))), (0,255,0), lineThickness)
    if(moveDiff>0):
        print("ENTER")
    else:
        print("EXIT")
    return image,cross



def drawCount(img,s):
    # Write some Text
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,200)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    
    cv2.putText(img,s, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    return img


def enterOrExit(points):
    moveDiff=0
    for i in range(1,len(points)):
        moveDiff=moveDiff+(points[i][0]-points[i-1][0])
        print((points[i-1][0] , points[i-1][1]), (points[i][0], points[i][1]))
    print(moveDiff)
    if(moveDiff>0):
        return (0,1)

        print("EXIT")
    else:
        print("ENTER")
        return (1,0)
    return 0,0
        
        
    
    

def position_link(new_position):

    isInserted=0
    for i in range( len(person_position)):

            
        (x, y, w, h,diff)=union(person_position[i][len(person_position[i])-1],new_position)
        print((x, y, w, h,diff))
        if(w>0 and h>0):
            person_position[i].extend([new_position])
            isInserted=1
            
#        if(diff>ALLOWED_RECOGNITION_MISSING_FRAME):
#            (Enter1,Exit1)=enterOrExit(person_position[i])
#            ENTER =ENTER+Enter1
#            EXIT =EXIT+Exit1
#            person_position.pop(i)
            
    if(isInserted==0):

        person_position.append([new_position])




def syn(count,ENTER,EXIT):  
    for i in range( len(person_position)):
        diff=abs(count-person_position[i][len(person_position[i])-1][4])
        if(diff>ALLOWED_RECOGNITION_MISSING_FRAME):
            (Enter1,Exit1)=enterOrExit(person_position[i])
            ENTER =ENTER+Enter1
            EXIT =EXIT+Exit1
            person_position.pop(i)
            
    return (ENTER,EXIT)
        
        
            
        
    
    
#cascade_src = 'D:/C-Drive/opencv3.4.3/build/etc/haarcascades/haarcascade_upperbody.xml'
##cascade_src = 'D:/C-Drive/opencv3.4.3/build/etc/haarcascades/haarcascade_fullbody.xml'
#video_src = 'D:/PROJECTS/Python/HUMAN COUNT/dataset/VID_20191029_185101.mp4'
#video_src = 'E:/Google Drive/PCDS/106_20150509_back/noisy/uncrowd/2015_05_09_15_05_40BackColor.avi'


cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

pre_rect=[(0,0,0,0)]
global  ENTER 
global  EXIT 
person_position=[]
ENTER =0
EXIT=0



count=0
while True:
    ret, img = cap.read()
    img = image_resize(img, height = 200)
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
        position_link((x,y,w,h,count))
#        xxx,yyy,www,hhh,dddd=union(pre_rect[len(pre_rect)-1],(x,y,w,h,count))
#        pre_rect.append((x,y,w,h,dddd))
#        img,cross=drawLines(pre_rect,img)
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) 
    ENTER,EXIT =syn(count,ENTER,EXIT)
#    cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)      
    img=drawCount(img,"Enter:"+str(ENTER)+", Exit:"+str(EXIT))
    cv2.imshow('img', img)
    count+=1
    
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()