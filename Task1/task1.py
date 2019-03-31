
#Name: Shreyas Narasimha
#UBID: sn58
#UB Person Number: 50289736

import cv2
import numpy as np
import math

w = 0
h = 0
   
#dimensions are known by using img.shape()
img = cv2.imread("task1.png", 0)
for i in img:
    h+=1
    
for i in img[0]:
    w+=1  

filtered = []
nlist = [[0 for j in range(w)] for i in range(h)]
xtemplist = [[0 for j in range(w)] for i in range(h)]
ytemplist = [[0 for j in range(w)] for i in range(h)]
maggi = [[0 for j in range(w)] for i in range(h)]
oreo =  [[0 for j in range(w)] for i in range(h)]

def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def norm(edge,w,h):
    m = 0.0
    for i in range(h):
        for j in range(w):
            if(abs(edge[i][j]) > m):
                m=abs(edge[i][j])
    edge = [[abs(item)/m for item in subl] for subl in edge]
    return np.asarray(edge, dtype = np.float32)            
                
            
    
for i in range(h):
    for j in range(w):
        nlist[i][j] = img[i][j]
        
edge_x = [[0 for j in range(w)] for i in range(h)]
edge_y = [[0 for j in range(w)] for i in range(h)]

# Computing vertical edges
#We manually write the equivalent of te above
#Aim is to write the code above manually without using OpenCV for vertical edges
for i in range(1,h-1):
    for j in range(1,w-1):
        xtemplist[i][j] = nlist[i-1][j+1] + 2*nlist[i][j+1] + nlist[i+1][j+1] - nlist[i+1][j-1] - 2*nlist[i][j-1] - nlist[i+1][j-1]
        ytemplist[i][j] = nlist[i+1][j-1] + 2*nlist[i+1][j] + nlist[i+1][j+1] - nlist[i-1][j-1] - 2*nlist[i-1][j] - nlist[i-1][j+1]
        #maggi[i][j] = math.sqrt(xtemplist[i][j]*xtemplist[i][j] + ytemplist[i][j]*ytemplist[i][j])
        #oreo[i][j] = math.atan2(ytemplist[i][j] , xtemplist[i][j] + 1e-3)
#Computations
edge_x = np.asarray(xtemplist, dtype = np.float32) #Vertical response
edge_y = np.asarray(ytemplist, dtype = np.float32) #Horizontal response
'''
# magnitude of edges (combining horizontal and vertical edges)
a = max(maggi)
maggi = [[item/max(a) for item in subl] for subl in maggi]
edge_magnitude = np.asarray(maggi, dtype=np.float32)

#Orientation of edges
oreo = [[item * 180/3.14 for item in subl] for subl in oreo]
b = max(oreo)
oreo = [[item/max(b) for item in subl] for subl in oreo]
edge_direction = np.asarray(oreo, dtype = np.float32)
'''
#Display0 - original image

#cv2.imwrite('edge_x.jpg',edge_x)
cv2.imwrite('norm_x.jpg',norm(edge_x,w,h))
#cv2.imwrite('edge_y.jpg',edge_y)
cv2.imwrite('norm_y.jpg',norm(edge_y,w,h))

display('original',img)
display('edge_x',edge_x)
display('norm_x',norm(edge_x,w,h))
display('edge_y',edge_y)
display('norm_y',norm(edge_y,w,h))
'''
#Display3 - magnitude
display('edge_magnitude', edge_magnitude)

#Display4 - Direction
display('edge_direction', edge_direction)

'''


#print("Original image size: {:4d} x {:4d}".format(img.shape[0], img.shape[1]))
#print("Resulting image size: {:4d} x {:4d}".format(edge_magnitude.shape[0], edge_magnitude.shape[1]))
