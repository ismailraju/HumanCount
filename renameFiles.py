import os

inputpath = 'E:/PROJECT ALL/kaggle/project/human Count/HumanCountraju/haarcascade/p/'
 



for filename in os.listdir(inputpath): 
#        dst ="Hostel" + str(i) + ".jpg"
        src =inputpath+ filename 

        filename1=filename.replace("(", "", 10)
        filename1=filename1.replace(")", "", 10)
        filename1=filename1.replace(" ", "", 10)
        print(filename1)
        dst =inputpath+ filename1 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 