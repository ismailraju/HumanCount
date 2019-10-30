 
import cv2
 

cascade_src = 'E:/PROJECT ALL/kaggle/project/human Count/HumanCountraju/haarcascade/classifier/cascade.xml'

#cascade_src = 'D:/C-Drive/opencv3.4.3/build/etc/haarcascades/haarcascade_fullbody.xml'
video_src = 'E:/PROJECT ALL/kaggle/project/human Count/dataset/Project 1.avi'
#video_src = 'E:/Google Drive/PCDS/106_20150509_back/noisy/uncrowd/2015_05_09_15_05_40BackColor.avi'

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
  return (x, y, w, h)


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
#cascade_src = 'D:/C-Drive/opencv3.4.3/build/etc/haarcascades/haarcascade_upperbody.xml'
##cascade_src = 'D:/C-Drive/opencv3.4.3/build/etc/haarcascades/haarcascade_fullbody.xml'
#video_src = 'D:/PROJECTS/Python/HUMAN COUNT/dataset/VID_20191029_185101.mp4'
#video_src = 'E:/Google Drive/PCDS/106_20150509_back/noisy/uncrowd/2015_05_09_15_05_40BackColor.avi'


cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

pre_rect=[(0,0,0,0)]


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
        xxx,yyy,www,hhh=union(pre_rect[len(pre_rect)-1],(x,y,w,h))
        pre_rect.append((x,y,w,h))
        img,cross=drawLines(pre_rect,img)
        img=drawCount(img,"Enter:"+str(cross)+", Exit:0")
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
#    cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)      
    
    cv2.imshow('img', img)
    
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()