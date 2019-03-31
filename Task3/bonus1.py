import cv2
import numpy as np
from matplotlib import pyplot as plt
def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



template1 = cv2.imread('a1.png',0)
template1 = cv2.Laplacian(template1,cv2.CV_8U,5)
template1 = cv2.convertScaleAbs(template1)
template2 = cv2.imread('a2.png',0)
template2 = cv2.Laplacian(template2,cv2.CV_8U,5)
template2 = cv2.convertScaleAbs(template2)
template3 = cv2.imread('a3.png',0)
template3 = cv2.Laplacian(template3,cv2.CV_8U,5)
template3 = cv2.convertScaleAbs(template3)
template4 = cv2.imread('a4.png',0)
template4 = cv2.Laplacian(template4,cv2.CV_8U,5)
template4 = cv2.convertScaleAbs(template4)
template5 = cv2.imread('a5.png',0)
template5 = cv2.Laplacian(template5,cv2.CV_8U,5)
template5 = cv2.convertScaleAbs(template5)
template6 = cv2.imread('a6.png',0)
template6 = cv2.Laplacian(template6,cv2.CV_8U,5)
template6 = cv2.convertScaleAbs(template6)
template7 = cv2.imread('a7.png',0)
template7 = cv2.Laplacian(template7,cv2.CV_8U,5)
template7 = cv2.convertScaleAbs(template7)

w = 25 
h = 25    
res = []



threshold = 0.63
for i in range(1,16):
        img = cv2.imread('pos_'+str(i)+'.jpg')
        img1 = img.copy()
        img = cv2.GaussianBlur(img,(3,3),0)        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
        img = cv2.Laplacian(img,cv2.CV_8U,5)        
        img = cv2.convertScaleAbs(img)
        res = cv2.matchTemplate(img,template1,cv2.TM_CCORR_NORMED)
        '''
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img1, pt, (pt[0] + w, pt[1] + h), (0,0,255 ), 2)
        display('res.png',img1)
        '''
        res1 = cv2.matchTemplate(img,template1,cv2.TM_CCORR_NORMED)        
        loc1 = np.where( res1 >= threshold)
        for pt in zip(*loc1[::-1]):
            cv2.rectangle(img1, pt, (pt[0] + 23, pt[1] + 23), (255,0,0 ), 2)
        #display('res.png',img1)
      
        res2 = cv2.matchTemplate(img,template2,cv2.TM_CCORR_NORMED)        
        loc2 = np.where( res2 >= threshold)
        for pt in zip(*loc2[::-1]):
            cv2.rectangle(img1, pt, (pt[0] + 21, pt[1] + 21), (255,0,0 ), 2)
        #display('res.png',img1)    
        
        res3 = cv2.matchTemplate(img,template3,cv2.TM_CCORR_NORMED)        
        loc3 = np.where( res3 >= threshold)
        for pt in zip(*loc3[::-1]):
            cv2.rectangle(img1, pt, (pt[0] + 25, pt[1] + 25), (255,0,0 ), 2)
        #display('res.png',img1)            
        
        res4 = cv2.matchTemplate(img,template4,cv2.TM_CCORR_NORMED)        
        loc4 = np.where( res4 >= threshold)
        for pt in zip(*loc4[::-1]):
            cv2.rectangle(img1, pt, (pt[0] + 16, pt[1] + 22), (255,0,0 ), 2)
       # display('res.png',img1)          
            
        
        res5 = cv2.matchTemplate(img,template5,cv2.TM_CCORR_NORMED)        
        loc5 = np.where( res5 >= 0.6)
        for pt in zip(*loc5[::-1]):
            cv2.rectangle(img1, pt, (pt[0] + 16, pt[1] + 16), (255,0,0 ), 2)
       
        res6 = cv2.matchTemplate(img,template6,cv2.TM_CCORR_NORMED)        
        loc6 = np.where( res6 >= threshold)
        for pt in zip(*loc6[::-1]):
            cv2.rectangle(img1, pt, (pt[0] + 17, pt[1] + 17 ), (255,0,0 ), 2)
        
        
        res7 = cv2.matchTemplate(img,template7,cv2.TM_CCORR_NORMED)        
        loc7 = np.where( res7 >= threshold)
        for pt in zip(*loc7[::-1]):            
            cv2.rectangle(img1, pt, (pt[0] + 17, pt[1] + 17 ), (255,0,0 ), 2)
        
        display('res.png',img1)
        

#template = cv2.resize(template,(9, 15), interpolation = cv2.INTER_CUBIC)
#img = cv2.Canny(img,0,250)
#img = cv2.convertScaleAbs(img)
#template = cv2.Canny(template,0,250)
#img = cv2.bitwise_not(img)