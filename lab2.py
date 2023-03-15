import cv2
import sys
import scipy
import numpy as np
from matplotlib import pyplot as plt

def HarrisPointsDetector(img, window_size, threshold = 50):
    offset = int((window_size-1)/2)
    # Find derivatives
    X = cv2.Sobel(src=img, ddepth=-1, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Y = cv2.Sobel(src=img, ddepth=-1, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT)
    XY = cv2.Sobel(src=img, ddepth=-1, dx=1, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT)
    
    # Apply gaussian
    gXX = cv2.GaussianBlur(X**2, ksize=(5,5), sigmaX=0.5).astype(np.float32)
    gYY = cv2.GaussianBlur(Y**2, ksize=(5,5), sigmaX=0.5).astype(np.float32)
    gXY = cv2.GaussianBlur(XY, ksize=(5,5), sigmaX=0.5).astype(np.float32)

    # Compute gradient orientation for each pixel
    orientation = np.arctan2(Y, X) * (180/np.pi)

    # Calculate Harris response
    harrisResponse = gXX * gYY - gXY**2 - 0.05 * (gXX+gYY)**2

    # Get responses > 0
    harrisResponse[harrisResponse < 0] = 0

    # Normalise 0 - 255
    harrisResponse = (255*(harrisResponse - np.min(harrisResponse))/np.ptp(harrisResponse))

    points, harrisResponse = NMS(harrisResponse, 7)
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

def main():
    # Open image
    # filename = 'images/bernieSanders.jpg'
    filename = "images/grey.jpg"
    # filename = "images/800px-Checkerboard_pattern.svg.png"
    img = cv2.imread(filename)

    # Check for success
    if img is None:
        print('Error: failed to open', filename)
        sys.exit()

    # Convert to greyscale and save it
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur src image
    blur = cv2.GaussianBlur(grey, ksize=(3,3), sigmaX=2)

    features, responses = HarrisPointsDetector(blur, 2)

    threshold_count = [len(responses[responses > i]) for i in range(256)]
    
    # print(responses[0:100])
    t = plt.figure(1)
    output = cv2.drawKeypoints(img, features, None, color=(0,255,0), flags=0)
    plt.imshow(output)

    f = plt.figure(2)
    plt.plot(threshold_count)
    plt.xlabel("Threshold")
    plt.ylabel("Number of Interest Points")

    plt.show()


if __name__ == "__main__":
    main()


