

import cv2
import numpy as np
from matplotlib import pyplot as plt
def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
for i in range(1,16):
    img = cv2.imread('pos_'+str(i)+'.jpg')
    #img = cv2.imread('neg_'+str(i)+'.jpg')
    #line for testing negative images
    #img = cv2.imread('str(i)+'.jpg')
    #line for testing if all files are only numbered
    img = cv2.GaussianBlur(img,(3,3),0)
    img1 = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    template = cv2.imread('4.png',0)
    w, h = template.shape[::-1]
    img = cv2.Laplacian(img,cv2.CV_8U,5)
    template = cv2.Laplacian(template,cv2.CV_8U,5)
    img = cv2.convertScaleAbs(img)
    template = cv2.convertScaleAbs(template)
    res = cv2.matchTemplate(img,template,cv2.TM_CCORR_NORMED)
    threshold = 0.62
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img1, pt, (pt[0] + w, pt[1] + h), (0,0,255 ), 2)
    display('res.png',img1)


display('',template)

#template = cv2.resize(template,(9, 15), interpolation = cv2.INTER_CUBIC)
#img = cv2.Canny(img,0,250)
#img = cv2.convertScaleAbs(img)
#template = cv2.Canny(template,0,250)
#img = cv2.bitwise_not(img)