import cv2
import numpy as np

#function so that the np array of 4 points is in top-left, top-right, bottom right, bottom-left order
def order_points(pts):
    pts = pts.reshape((4,2))
    rect = np.zeros((4,2),dtype = np.float32)

    add = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(add)]
    rect[2] = pts[np.argmax(add)]

    diff = np.diff(pts,axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# function returns a list of all edges/contours in an image
def findEdges(image):    
    img_gauss = cv2.GaussianBlur(image,(5,5),0)  #blurring using a 5x5 matrix to reduce noise

    kernel = np.ones((5,5),np.uint8)
    img_gauss_open = cv2.morphologyEx(img_gauss, cv2.MORPH_OPEN, kernel) #opening performs erosion followed by dilation, to further reduce noise

    edged = cv2.Canny(img_gauss_open,30,50)  #applying canny algorithm to return edges of an image

    contours, _ = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #return the contours as a list, with simple apprximation model
    return contours

# function returns a numpy array of the 4 vertices of the largest rectangle in the image
def largestRect(contours):
    contours = sorted(contours,key = cv2.contourArea,reverse=True) #sort in descending order of area of contour

    #the loop finds the boundary of the document 
    for contour in contours:
        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,0.04 * peri,True) #this function reduces contour to simple points

        # the first contour with 4 vertices is the document's boundary as it has the largest area of all rectangles
        if len(approx) == 4 :
            break
    approx = order_points(approx)
    return approx 

img = cv2.imread('/Users/yuvanseth/Documents/VS/NLP_OCR/document_basic.png', 0)   # read the image and create its grayscale instance
width = 500
height = 750
img = cv2.resize(img,(width,height)) 

contours = findEdges(img)
approx = largestRect(contours)

points = np.float32([[0,0],[width,0],[width,height],[0,height]])  # map to width x height target window

matrix = cv2.getPerspectiveTransform(approx,points)  
persp = cv2.warpPerspective(img,matrix,(width,height)) 

scanned = cv2.adaptiveThreshold(persp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 10) 

cv2.imshow("Scanned",scanned)

cv2.waitKey(0)
cv2.destroyAllWindows()