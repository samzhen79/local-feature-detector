import cv2
import sys
import scipy
import numpy as np
from matplotlib import pyplot as plt

def threshold(img, T):
    return np.where(img > T, 255, 0)

# Open image
filename = 'images/bernieSanders.jpg'
img = cv2.imread(filename)

# Check for success
if img is None:
    print('Error: failed to open', filename)
    sys.exit()

# Convert to greyscale and save it
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur src image
blur = cv2.GaussianBlur(grey, ksize=(3,3), sigmaX=2)

# Find derivatives
X = cv2.Sobel(src=blur, ddepth=-1, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REFLECT)
Y = cv2.Sobel(src=blur, ddepth=-1, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT)
XY = cv2.Sobel(src=blur, ddepth=-1, dx=1, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT)

# Apply gaussian
gXX = cv2.GaussianBlur(X**2, ksize=(5,5), sigmaX=0.5)
gYY = cv2.GaussianBlur(Y**2, ksize=(5,5), sigmaX=0.5)
gXY = cv2.GaussianBlur(X*Y, ksize=(5,5), sigmaX=0.5)

# Second Moment Matrix for each point
# Kind of takes a while preferably skip this and just compute Harris response immediately instead
points = [[[[gXX[x][y], gXY[x][y]], [gXY[x][y], gYY[x][y]]] for y in range(len(gXX[x]))] for x in range(len(gXX))]
orientation = [[(np.arctan2(Y[x][y], X[x][y]) * (180/np.pi)) for y in range(len(Y[x]))] for x in range(len(Y))]

# Calculate Harris response for each matrix
harrisResponse =  [[np.linalg.det(M) - 0.05 *(np.trace(M))**2 for M in row] for row in points]

interestPoints = [[response if response > 0 else 0 for response in row] for row in harrisResponse]

def addBorder(img, pixels):
    output = np.zeros((img.shape[0]+pixels+(pixels*1),img.shape[1]+pixels+(pixels*1)))
    output[pixels:-pixels, pixels:-pixels] = img
    return output

def NMS(interestPoints, size):
    offset = int((size-1)/2)
    interestPoints = addBorder(np.array(interestPoints), 3)
    # suppression = []
    # for x in range(offset, len(interestPoints)-offset):
    #     for y in range(offset, len(interestPoints[x])-offset):
    #         max = max(list(np.concatenate(interestPoints[x-offset:x+offset+1, y-offset:y+offset+1]).flat))
    #         if (interestPoints[x][y] == max):
    #             suppression.append([x,y])
    suppression = [[[x,y] for y in range(offset, len(interestPoints[x])-offset) if (interestPoints[x][y] == max(list(np.concatenate(interestPoints[x-offset:x+offset+1, y-offset:y+offset+1]).flat)))] for x in range(offset, len(interestPoints)-offset)]
    suppression = list(np.concatenate(suppression).flat)
    return suppression

