
#Name: Shreyas Narasimha
#UBID: sn58
#UB Person Number: 50289736

import cv2
import numpy as np
import math
import time
s = time.time()
#declarations
#convert numpy array to a nested list

h = 0
w = 0
#dimensions are known by using img.shape()
img = cv2.imread("task2.jpg", 0)
for i in img:
    h+=1
    
for i in img[0]:
    w+=1  

nlist = [[0 for j in range(w+6)] for i in range(h+6)]
for i in range(h):
    for j in range(w):
        nlist[i+3][j+3] = img[i][j]
      
        
#code to create a list
def padlist(img,w,h):
    w = int(w/2)
    h = int(h/2)
    temp = [[0 for i in range(w+6)] for j in range(h+6)]
    for i in range(h):
        for j in range(w):
            temp[i+3][j+3] = img[i][j]
    return temp        
#display function
def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def downsize(img,h,w,octno):    
    hn = int(h/2)
    wn = int(w/2)
    print(hn,wn)
    temp =[[0 for j in range(wn)] for i in range(hn)]
    for i in range(hn):
        for j in range(wn):
            temp[i][j] = img[i*octno][j*octno]       
    return np.asarray(temp, dtype = np.uint8)        
    
#function to calculate g(x,y)    
def transform(a,sigma,w,h):
    w = int(w)
    h = int(h)
    t = 0.0
    xtemplist = [[0 for j in range(w)] for i in range(h)]
    b = [[0 for j in range(7)] for i in range(7)]
    for i in range(-3,4):
        for j in range(-3,4):
            b[i+3][j+3] = (1/(2*math.pi*(sigma**2)))*(math.exp(-(i**2 + j**2)/(2*sigma**2)))
            t+=b[i+3][j+3]
    b = [[item/t for item in subl]for subl in b]        
    for i in range(h):
        for j in range(w):
            xtemplist[i][j] = (a[i][j] * b[0][0]) + (a[i][j+1] * b[0][1]) + (a[i][j+2] * b[0][2]) + (a[i][j+3] * b[0][3]) + (a[i][j+4] * b[0][4]) + (a[i][j+5] * b[0][5]) + (a[i][j+6] * b[0][6]) + (a[i+1][j] * b[1][0]) + (a[i+1][j+1] * b[1][1]) + (a[i+1][j+2] * b[1][2]) + (a[i+1][j+3] * b[1][3]) + (a[i+1][j+4] * b[1][4]) + (a[i+1][j+5] * b[1][5]) + (a[i+1][j+6] * b[1][6]) + (a[i+2][j] * b[2][0]) + (a[i+2][j+1] * b[2][1]) + (a[i+2][j+2] * b[2][2]) + (a[i+2][j+3] * b[2][3]) + (a[i+2][j+4] * b[2][4]) + (a[i+2][j+5] * b[2][5]) + (a[i+2][j+6] * b[2][6]) + (a[i+3][j] * b[3][0]) + (a[i+3][j+1] * b[3][1]) + (a[i+3][j+2] * b[3][2]) + (a[i+3][j+3] * b[3][3]) + (a[i+3][j+4] * b[3][4]) + (a[i+3][j+5] * b[3][5]) + (a[i+3][j+6] * b[3][6]) + (a[i+4][j] * b[4][0]) + (a[i+4][j+1] * b[4][1]) + (a[i+4][j+2] * b[4][2]) + (a[i+4][j+3] * b[4][3]) + (a[i+4][j+4] * b[4][4]) + (a[i+4][j+5] * b[4][5]) + (a[i+4][j+6] * b[4][6]) + (a[i+5][j] * b[5][0]) + (a[i+5][j+1] * b[5][1]) + (a[i+5][j+2] * b[5][2]) + (a[i+5][j+3] * b[5][3]) + (a[i+5][j+4] * b[5][4]) + (a[i+5][j+5] * b[5][5]) + (a[i+5][j+6] * b[5][6]) + (a[i+6][j] * b[6][0]) + (a[i+6][j+1] * b[6][1]) + (a[i+6][j+2] * b[6][2]) + (a[i+6][j+3] * b[6][3]) + (a[i+6][j+4] * b[6][4]) + (a[i+6][j+5] * b[6][5]) + (a[i+6][j+6] * b[6][6])        
    return np.asarray(xtemplist, dtype = np.float32)
    #return xtemplist

def laplace(img1,img2,w,h):
    w = int(w)
    h = int(h)
    m = 0
    temp3 = [[0 for i in range(w)] for j in range(h)]
    for i in range(h):
        for j in range(w):
            temp3[i][j] = (img1[i][j] - img2[i][j])        
    temp3 = np.asarray(temp3, dtype = np.float32)
    return temp3

def keypoints(l1,l2,l3,l4,w,h,n):
    w = int(w)
    h = int(h)
    temp1 = [[0 for i in range(w)] for j in range(h)]
    temp2 = [[0 for i in range(w)] for j in range(h)]
    for i in range(1,h-1):
        for j in range(1,w-1):
            if((l2[i][j] > max(l1[i-1][j-1],l1[i-1][j],l1[i-1][j+1],l1[i][j-1],l1[i][j],l1[i][j+1],l1[i+1][j-1],l1[i+1][j],l1[i+1][j+1],l2[i-1][j-1],l2[i-1][j],l2[i-1][j+1],l2[i][j-1],l2[i][j+1],l2[i+1][j-1],l2[i+1][j],l2[i+1][j+1],l3[i-1][j-1],l3[i-1][j],l3[i-1][j+1],l3[i][j-1],l3[i][j],l3[i][j+1],l3[i+1][j-1],l3[i+1][j],l3[i+1][j+1])) or (l2[i][j] < min(l1[i-1][j-1],l1[i-1][j],l1[i-1][j+1],l1[i][j-1],l1[i][j],l1[i][j+1],l1[i+1][j-1],l1[i+1][j],l1[i+1][j+1],l2[i-1][j-1],l2[i-1][j],l2[i-1][j+1],l2[i][j-1],l2[i][j+1],l2[i+1][j-1],l2[i+1][j],l2[i+1][j+1],l3[i-1][j-1],l3[i-1][j],l3[i-1][j+1],l3[i][j-1],l3[i][j],l3[i][j+1],l3[i+1][j-1],l3[i+1][j],l3[i+1][j+1] ))):
                if(l2[i][j]>n):
                    temp1[i][j] = 255
                else:
                    temp1[i][j] = 0
    for i in range(1,h-1):
        for j in range(1,w-1):
            if((l3[i][j] > max(l2[i-1][j-1],l2[i-1][j],l2[i-1][j+1],l2[i][j-1],l2[i][j],l2[i][j+1],l2[i+1][j-1],l2[i+1][j],l2[i+1][j+1],l3[i-1][j-1],l3[i-1][j],l3[i-1][j+1],l3[i][j-1],l3[i][j+1],l3[i+1][j-1],l3[i+1][j],l3[i+1][j+1],l4[i-1][j-1],l4[i-1][j],l4[i-1][j+1],l4[i][j-1],l4[i][j],l4[i][j+1],l4[i+1][j-1],l4[i+1][j],l4[i+1][j+1])) or (l3[i][j] < min(l2[i-1][j-1],l2[i-1][j],l2[i-1][j+1],l2[i][j-1],l2[i][j],l2[i][j+1],l2[i+1][j-1],l2[i+1][j],l2[i+1][j+1],l3[i-1][j-1],l3[i-1][j],l3[i-1][j+1],l3[i][j-1],l3[i][j+1],l3[i+1][j-1],l3[i+1][j],l3[i+1][j+1],l4[i-1][j-1],l4[i-1][j],l4[i-1][j+1],l4[i][j-1],l4[i][j],l4[i][j+1],l4[i+1][j-1],l4[i+1][j],l4[i+1][j+1] ))):
                if(l3[i][j]>n):
                    temp2[i][j] = 255
                else:
                    temp2[i][j] = 0
    return (np.asarray(temp1, dtype=np.uint8), np.asarray(temp2, dtype=np.uint8))  

def overlay(img,key1,key2,w,h):
    w = int(w)
    h = int(h)
    temp = [[0 for i in range(w)] for j in range(h)]
    for i in range(h):
        for j in range(w):
            if((key1[i][j] == 255) or (key2[i][j] == 255) ):
                temp[i][j] = 255
            else:
                temp[i][j] = img[i][j]
    return np.asarray(temp, dtype = np.uint8)

def finalkey(key,b,c,d,w,h):
    w1 = int(w/2)
    h1 = int(h/2)
    w2 = int(w1/2)
    h2 = int(h1/2)
    w3 = int(w2/2)
    h3 = int(h2/2)
    for i in range(h1):
        for j in range(w1):
            if(b[i][j] == 255):
                key[i*2][j*2] = 255
    for i in range(h2):
        for j in range(w2):
            if(b[i][j] == 255):
                key[i*4][j*4] = 255
    for i in range(h3):
        for j in range(w3):
            if(b[i][j] == 255):
                key[i*8][j*8] = 255
    return np.asarray(key, dtype= np.uint8)
            
                

#1st oct
a1 = transform(nlist,(1/math.sqrt(2)),w,h)
a2 = transform(nlist,1,w,h)
a3 = transform(nlist,(math.sqrt(2)),w,h)
a4 = transform(nlist,2,w,h)
a5 = transform(nlist,(2*math.sqrt(2)),w,h)
al1 = laplace(a1,a2,w,h)
al2 = laplace(a2,a3,w,h)
al3 = laplace(a3,a4,w,h)
al4 = laplace(a4,a5,w,h)
(ak1,ak2)  = keypoints(al1,al2,al3,al4,w,h,5)
af = overlay(img, ak1,ak2,w,h)

#2nd Oct
b   = downsize(img,h,w,2)
bp  = padlist(b,w,h)
b1  = transform(bp,math.sqrt(2),w/2,h/2)
b2  = transform(bp,2,w/2,h/2)
b3  = transform(bp,(2*math.sqrt(2)),w/2,h/2)
b4  = transform(bp,4,w/2,h/2)
b5  = transform(bp,(4*math.sqrt(2)),w/2,h/2)
bl1 = laplace(b1,b2,w/2,h/2)
bl2 = laplace(b2,b3,w/2,h/2)
bl3 = laplace(b3,b4,w/2,h/2)
bl4 = laplace(b4,b5,w/2,h/2)
(bk1,bk2)  = keypoints(bl1,bl2,bl3,bl4,w/2,h/2,0.2)
bf = overlay(b, bk1,bk2,w/2,h/2)

#3rd oct

c   = downsize(img,h/2,w/2,4)
cp  = padlist(c,w/2,h/2)
c1  = transform(cp,2*math.sqrt(2),w/4,h/4)
c2  = transform(cp,4,w/4,h/4)
c3  = transform(cp,4*math.sqrt(2),w/4,h/4)
c4  = transform(cp,8,w/4,h/4)
c5  = transform(cp,8*math.sqrt(2),w/4,h/4)
cl1 = laplace(c1,c2,w/4,h/4)
cl2 = laplace(c2,c3,w/4,h/4)
cl3 = laplace(c3,c4,w/4,h/4)
cl4 = laplace(c4,c5,w/4,h/4)
(ck1,ck2)  = keypoints(cl1,cl2,cl3,cl4,w/4,h/4,0)
cf = overlay(c, ck1,ck2,w/4,h/4)


#4th Octave
d   = downsize(img,h/4,w/4,8)
dp  = padlist(d,w/4,h/4)
d1  = transform(dp,4*math.sqrt(2),w/8,h/8)
d2  = transform(dp,8,w/8,h/8)
d3  = transform(dp,8*math.sqrt(2),w/8,h/8)
d4  = transform(dp,16,w/8,h/8)
d5  = transform(dp,16*math.sqrt(2),w/8,h/8)
dl1 = laplace(d1,d2,w/8,h/8)
dl2 = laplace(d2,d3,w/8,h/8)
dl3 = laplace(d3,d4,w/8,h/8)
dl4 = laplace(d4,d5,w/8,h/8)
(dk1,dk2)  = keypoints(dl1,dl2,dl3,dl4,w/8,h/8,0)
df = overlay(d, dk1,dk2,w/8,h/8)

#Calculating final keypoints by overlaying the downsized keypoints on the original image
keypointImage = finalkey(af,bf,cf,df,w,h)
count = 1
#to print left most point positions
for i in range(2,h):
    for j in range(5):
            if(keypointImage[i][j] == 255):
                print(i,j)
                count = count+1
t = time.time()
print(t-s)
#Writing data for report

cv2.imwrite('pics/Octave2.jpg',np.asarray(b, dtype = np.uint8))
cv2.imwrite('pics/Octave21.jpg',np.asarray(b1, dtype = np.uint8))
cv2.imwrite('pics/Octave22.jpg',np.asarray(b2, dtype = np.uint8))
cv2.imwrite('pics/Octave23.jpg',np.asarray(b3, dtype = np.uint8))
cv2.imwrite('pics/Octave24.jpg',np.asarray(b4, dtype = np.uint8))
cv2.imwrite('pics/Octave25.jpg',np.asarray(b5, dtype = np.uint8))
cv2.imwrite('pics/Octave3.jpg',np.asarray(c, dtype = np.uint8))
cv2.imwrite('pics/Octave31.jpg',np.asarray(c1, dtype = np.uint8))
cv2.imwrite('pics/Octave32.jpg',np.asarray(c2, dtype = np.uint8))
cv2.imwrite('pics/Octave33.jpg',np.asarray(c3, dtype = np.uint8))
cv2.imwrite('pics/Octave34.jpg',np.asarray(c4, dtype = np.uint8))
cv2.imwrite('pics/Octave35.jpg',np.asarray(c5, dtype = np.uint8))
cv2.imwrite('pics/DOG21.jpg',bl1)
cv2.imwrite('pics/DOG22.jpg',bl2)
cv2.imwrite('pics/DOG23.jpg',bl3)
cv2.imwrite('pics/DOG24.jpg',bl4)
cv2.imwrite('pics/DOG31.jpg',cl1)
cv2.imwrite('pics/DOG32.jpg',cl2)
cv2.imwrite('pics/DOG33.jpg',cl3)
cv2.imwrite('pics/DOG34.jpg',cl4)



#Display everything

display('1',img)
display('11',np.asarray(a1, dtype = np.uint8))
display('12',np.asarray(a2, dtype = np.uint8))
display('13',np.asarray(a3, dtype = np.uint8))
display('14',np.asarray(a4, dtype = np.uint8))
display('15',np.asarray(a5, dtype = np.uint8))
display('l11',al1)
#print(al1)
display('l12',al2)
display('l13',al3)
#print(al3)
display('l14',al4)
display('k11',ak1)
display('k12',ak2)
display('key',af)


display('2',np.asarray(b, dtype = np.uint8))
display('21',np.asarray(b1, dtype = np.uint8))
display('22',np.asarray(b2, dtype = np.uint8))
display('23',np.asarray(b3, dtype = np.uint8))
display('24',np.asarray(b4, dtype = np.uint8))
display('25',np.asarray(b5, dtype = np.uint8)) 
display('l21',bl1)
display('l22',bl2)
display('l23',bl3) 
display('l24',bl4)
display('k21',bk1)
display('k22',bk2)
display('key',bf)

display('3',np.asarray(c, dtype = np.uint8))
display('31',np.asarray(c1, dtype = np.uint8))
display('32',np.asarray(c2, dtype = np.uint8))
display('33',np.asarray(c3, dtype = np.uint8))
display('34',np.asarray(c4, dtype = np.uint8))
display('35',np.asarray(c5, dtype = np.uint8))
display('l31',cl1)  
display('l32',cl2)
display('l33',cl3)
display('l34',cl4)
display('k31',ck1)
display('k32',ck2)
display('key',cf)

display('4',np.asarray(d, dtype = np.uint8))
display('41',np.asarray(d1, dtype = np.uint8))
display('42',np.asarray(d2, dtype = np.uint8))
display('43',np.asarray(d3, dtype = np.uint8))
display('44',np.asarray(d4, dtype = np.uint8))
display('45',np.asarray(d5, dtype = np.uint8))
display('l41',dl1)
display('l42',dl2)
display('l43',dl3)
display('l44',dl4)
display('k41',dk1)
display('k42',dk2)
display('key',df)

display('final keypoints',keypointImage)


 


        

