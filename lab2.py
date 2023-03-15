import cv2
import sys
import scipy
import numpy as np
from matplotlib import pyplot as plt

def HarrisPointsDetector(img, threshold = None):
    # Find derivatives
    X = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Y = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT)
    XY = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT)
    
    XX = X**2
    YY = Y**2

    # Apply gaussian
    gXX = cv2.GaussianBlur(XX, ksize=(5,5), sigmaX=0.5).astype(np.float32)
    gYY = cv2.GaussianBlur(YY, ksize=(5,5), sigmaX=0.5).astype(np.float32)
    gXY = cv2.GaussianBlur(XY, ksize=(5,5), sigmaX=0.5).astype(np.float32)

    # Compute gradient orientation for each pixel
    orientation = np.arctan2(Y, X) * (180/np.pi)

    # Calculate Harris response
    harrisResponse = ((gXX * gYY) - (gXY)**2 - 0.05*((gXX+gYY)**2))

    # Get responses > 0
    harrisResponse[harrisResponse < 0] = 0

    # Normalise 0 - 1000
    harrisResponse = (harrisResponse * (1000/np.max(harrisResponse))).astype(np.int32)

    points, harrisResponse = NMS(harrisResponse, 7)
    if (threshold == None):
        threshold = 0.1 * np.max(harrisResponse)
    print("Done NMS")
    
    kp = []
    
    for i in range(len(points)):
        x, y = points[i][0], points[i][1]
        if (harrisResponse[x][y] > threshold):
            kp.append(cv2.KeyPoint(y, x, harrisResponse[x][y], orientation[x][y]))
    print("Done KP")
        
    return kp, harrisResponse

def addBorder(img, pixels):
    output = np.zeros((img.shape[0]+pixels+(pixels*1),img.shape[1]+pixels+(pixels*1)))
    output[pixels:-pixels, pixels:-pixels] = img
    return output

def NMS(interestPoints, size):
    offset = int((size-1)/2)
    interestPoints = addBorder(np.array(interestPoints), offset)
    suppression = []
    for x in range(offset, len(interestPoints)-offset):
        for y in range(offset, len(interestPoints[x])-offset):
                max_val = np.max(interestPoints[x-offset:x+offset+1, y-offset:y+offset+1])
                if (interestPoints[x][y] == max_val):
                    interestPoints[x-offset:x+offset+1, y-offset:y+offset+1] = 0
                    interestPoints[x][y] = max_val
                    suppression.append([x-offset,y-offset])
    interestPoints = interestPoints[offset:-offset, offset:-offset]
    return suppression, interestPoints

def featureDescriptor(img, features):
    orb = cv2.ORV_create()
    des = orb.compute(img, features)
    return(des)

def main():
    # Open image
    filename = 'images/bernieSanders.jpg'
    # filename = "images/grey.jpg"
    # filename = "images/800px-Checkerboard_pattern.svg.png"
    # filename = "images/phpicsL2c.png"
    img = cv2.imread(filename)

    # Check for success
    if img is None:
        print('Error: failed to open', filename)
        sys.exit()

    # Convert to greyscale and save it
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur src image
    blur = cv2.GaussianBlur(grey, ksize=(3,3), sigmaX=2)

    features, responses = HarrisPointsDetector(blur, 0)
    f = open("something.txt", "w")
    f.write(np.array2string(responses, threshold=sys.maxsize))
    f.close()
    print(np.max(responses))
    threshold_count = []
    labels = []
    split = 1
    for i in range(int(np.max(responses)/split)):
        threshold_count.append((responses > i*split).sum())
    
    print(threshold_count)
    # print(responses[0:100])
    t = plt.figure(1)
    output = cv2.drawKeypoints(grey, features, None, color=(0,255,0), flags=0)
    plt.imshow(output)

    f = plt.figure(2)
    plt.plot(np.array(range(len(threshold_count)))*split, threshold_count)
    plt.title("Number of Interest Points against Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Number of Interest Points")

    plt.show()


if __name__ == "__main__":
    main()


