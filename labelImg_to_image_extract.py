# -*- coding: utf-8 -*-

import cv2
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


print(cv2.__version__)

for ff in ['2015_05_09_09_11_49BackColor',
            '2015_05_09_09_16_08BackColor',
            '2015_05_09_09_18_42BackColor',
            '2015_05_09_09_36_02BackColor',
            '2015_05_09_10_51_27BackColor',
            '2015_05_09_11_04_26BackColor',
            '2015_05_09_11_06_42BackColor',
            '2015_05_09_11_51_30BackColor',
            '2015_05_09_11_53_25BackColor',
            '2015_05_09_11_56_20BackColor',
            '2015_05_09_12_06_31BackColor',
            '2015_05_09_13_24_15BackColor',
            '2015_05_09_13_35_16BackColor',
            '2015_05_09_14_20_56BackColor',
            '2015_05_09_14_21_59BackColor',
            '2015_05_09_14_24_52BackColor',
            '2015_05_09_14_28_33BackColor',
            '2015_05_09_15_05_40BackColor',
            '2015_05_09_15_18_38BackColor',
            '2015_05_09_15_34_59BackColor',
            '2015_05_09_16_34_10BackColor',
            '2015_05_09_17_16_47BackColor']:
    
    
    inputpath = 'E:/PROJECT ALL/kaggle/project/human Count/dataset/uncrowd_noisy/'+ff+'/'
    outputpath = 'E:/PROJECT ALL/kaggle/project/human Count/dataset/pp/crop/'
    try:
        os.stat(outputpath)
    except:
        os.mkdir(outputpath)
    
    for file in os.listdir(inputpath):
        if file.endswith(".xml"):
            print(file)
            fileName=file.split(".xml")[0]
            root = ET.parse(inputpath+'/'+file).getroot()
            realImagePath=item=root.find('path').text
            
            realimg = cv2.imread(realImagePath)
            
            count=0;
            for Variable in root.findall('object'):
                item=str(Variable.find('name').text)
                x1=int(Variable.find('bndbox/xmin').text)
                x2=int(Variable.find('bndbox/xmax').text)
                y1=int(Variable.find('bndbox/ymin').text)
                y2=int(Variable.find('bndbox/ymax').text)
                
                
                crop_img = realimg[y1:y2, x1:x2]
                crop_img=cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    #            cv2.imshow("cropped", crop_img)
                file_output_path = os.path.join(outputpath+"/" , item+ff+'-'+fileName+"-"+str(count)+'.jpg')
                directory = os.path.dirname(file_output_path)
                
                try:
                    os.stat(directory)
                except:
                    os.mkdir(directory)
                
                
                cv2.imwrite(file_output_path, crop_img)
                count+=1
    #            print(Variable.get('name'), Variable.text)
            
            
            
            
    #        xmlPath=inputpath+'/'+file
    #        print(xmlPath)
    #        xmldoc = minidom.parse(xmlPath)
    #        itemlist = xmldoc.getElementsByTagName('object')
    #        print(len(itemlist))
    #        print(itemlist[0].attributes['name'].value)
            
            
    #        for s in itemlist:
    #            print(s.attributes['name'].value)
            
    #        print(os.path.join("/mydir", file))
    
    
    #count=0
    #while True:
    #    ret, img = cap.read()
    #    ret, img = cap.read()
    #    if (type(img) == type(None)):
    #        break
    #    
    #    cv2.imwrite(os.path.join(outputpath , 'Car-8-'+str(count)+'.jpg'), img)
    ##    cv2.imwrite("frame%d.jpg" % count, image)
    #    count+=1
    #    
    #   

cv2.destroyAllWindows()