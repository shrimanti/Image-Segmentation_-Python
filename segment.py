
import numpy as np

import cv2

import matplotlib.pyplot as plt



# Mouse Callback Function

F = np.zeros((384,512))
B = np.zeros((384,512))
P = np.zeros((384,512))
h = np.zeros(256)
f = np.zeros(256)


s = 0 
x1=-1
y1=-1

img = cv2.imread('10.png', 0)


def segment_image():
	global img,h,g,f
	print " \n This is the Image ! "
	cv2.imshow('F Image', F)
	cv2.imshow('B Image', B)
	
	for i in range (1 , 384):
		for j in range (1, 512):
			if F[i,j] == 1:
				P[i,j] = 1
			elif B[i,j] == 1:
				P[i,j] = 0
			else:
				P[i,j] = 0.5
	
	
	cv2.imshow('P Image', P)
	
	for i in range (1, 384):
		for j in range (1, 512):
			v = img[i,j]
			h[v] = h[v]+1
			
	plt.plot(h)	
	
	g = cv2.getGaussianKernel( 31, 5)
	g = np.transpose(g)[0]
	
	f = np.convolve(h,g,'same')
	
	
	plt.plot(f)
	
	plt.show()
						
	cv2.waitKey()
		
		
	
def draw_rectangle(event,x,y,flags,param):
	global s, x1, y1, F, B
	
	
	if event == cv2.EVENT_LBUTTONDBLCLK:
		if s == 0:
			x1,y1 = x,y
			s = 1
			
		elif s==1 :
			s=2
			cv2.rectangle(img,(x1,y1),(x,y),(100,150,10),-1)
			F[min(y1,y):max(y1,y),min(x1,x):max(x1,x)] = 1
			
			
		elif s==2:
			x1, y1 = x, y
			s=3
			
		else:
			cv2.rectangle(img,(x1,y1),(x,y),(200,50,100),-1)
			B[min(y1,y):max(y1,y),min(x1,x):max(x1,x)] = 1
			s=0
			
	
	
	if event == cv2.EVENT_RBUTTONDBLCLK:
		segment_image()
			
			
 
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_rectangle)

			
while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
       break    	   
	   
cv2.destroyAllWindows()




	
	
