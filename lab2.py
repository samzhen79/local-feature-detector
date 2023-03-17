import cv2
import sys
import os
import numpy as np
from scipy.ndimage import maximum_filter
from matplotlib import pyplot as plt

def HarrisPointsDetector(img, threshold = 0.01):
    # Find derivatives
    X = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Y = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT)
    
    XX = X**2
    YY = Y**2
    XY = X * Y

    # Apply gaussian
    gXX = cv2.GaussianBlur(XX, ksize=(5,5), sigmaX=0.5).astype(np.float32)
    gYY = cv2.GaussianBlur(YY, ksize=(5,5), sigmaX=0.5).astype(np.float32)
    gXY = cv2.GaussianBlur(XY, ksize=(5,5), sigmaX=0.5).astype(np.float32)

    # Compute gradient orientation for each pixel
    orientation = np.arctan2(Y, X) * (180/np.pi)

    # Calculate Harris response
    harrisResponse = ((gXX * gYY) - (gXY)**2 - 0.05*((gXX+gYY)**2)).astype(np.float32)

    # Get responses > 0
    harrisResponse[harrisResponse < 0] = 0

    # Non Maxima Suppresion
    points, harrisResponse = NMS(harrisResponse, 7)
    threshold = threshold * np.max(harrisResponse)
    print("Done NMS")
    
    # Create Key Points
    kp = []
    for i in range(len(points)):
        x, y = points[i][0], points[i][1]
        if (harrisResponse[x][y] > threshold):
            kp.append(cv2.KeyPoint(float(y), float(x), harrisResponse[x][y], angle=orientation[x][y], response=harrisResponse[x][y]))
    print("Done KP")
        
    return kp, harrisResponse

def addBorder(img, pixels):
    output = np.zeros((img.shape[0]+pixels+(pixels*1),img.shape[1]+pixels+(pixels*1)))
    output[pixels:-pixels, pixels:-pixels] = img
    return output

def NMS(interestPoints, size):
    # Create a maximum filter mask
    max_mask = maximum_filter(interestPoints, size)

    # Find local maxima
    local_maxima = (interestPoints == max_mask)

    # Get the coordinates of the local maxima
    maxima_coords = np.argwhere(local_maxima)

    # Create a new array to store the updated harrisResponse with local maxima only
    updated_harrisResponse = np.zeros_like(interestPoints)
    for coord in maxima_coords:
        updated_harrisResponse[coord[0], coord[1]] = interestPoints[coord[0], coord[1]]

    return maxima_coords, updated_harrisResponse

def featureDescriptor(img, features):
    orb = cv2.ORB_create()
    kp, des = orb.compute(img, features)
    return(des)

def SSDFeatureMatcher(des1, des2):
    if (des2 is None):
        return []
    matches = []
    for i in range(len(des1)):
        min_dist = float("inf")
        min_idx = -1
        for j in range(len(des2)):
            dist = float(np.sum((des1[i]-des2[j])**2))
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        matches.append(cv2.DMatch(i, min_idx, min_dist))
    return matches

def RatioFeatureMatcher(des1,des2):
    if (des2 is None):
        return []
    if len(des2) < 2:
        return []
    matches = []
    for i in range(len(des1)):
        dist = []
        for j in range(len(des2)):
            dist.append(float(np.sum((des1[i]-des2[j])**2)))
        min_index = np.argpartition(dist, min(1, len(dist) - 1))[:2]
        ratio = dist[min_index[0]]/(dist[min_index[1]]+0.00000001) # Avoid /0
        matches.append(cv2.DMatch(i, min_index[0], ratio))
    return matches

def getImg(filename):
    img = cv2.imread(filename)

    # Check for success
    if img is None:
        print('Error: failed to open', filename)
        sys.exit()

    return img

def main():
    # Open image
    base = 'images/bernieSanders.jpg'
    # bernie1 = "images/bernie180.jpg"
    # filename = "images/grey.jpg"
    # filename = "images/800px-Checkerboard_pattern.svg.png"
    # base = "images/phpicsL2c.png"
    
    # Get all images from folder
    images = []
    names = []
    for filename in os.listdir("images"):
        img = getImg(os.path.join("images",filename))
        images.append(img)
        names.append(filename)

    # Convert to greyscale and blur benchmark image
    base = getImg(base)
    control = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    control_blur = cv2.GaussianBlur(control, ksize=(3,3), sigmaX=2)

    # Do the same for all images in folder
    print("image prep")
    images_blur = []
    for image in images:
        temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_blur.append(cv2.GaussianBlur(temp, ksize=(3,3), sigmaX=2))

    # Detect corners
    print("detect corners")
    control_features, responses = HarrisPointsDetector(control_blur)
    test_features= []
    for i in range(len(images)):
        test_features.append(HarrisPointsDetector(images_blur[i])[0])

    print(np.max(responses))

    # Compute descriptors
    print("compute descriptors")
    des1 = featureDescriptor(control, control_features)
    descriptors = []
    for i in range(len(images)):
        descriptors.append(featureDescriptor(images[i], test_features[i]))

    print("Start Matching")
    matches = []
    for i in range(len(images)):
        matches.append(RatioFeatureMatcher(des1, descriptors[i]))

    ssd_matches = SSDFeatureMatcher(des1, descriptors[0])

    # Threshold distance
    print("Threshold distance")
    for i in range(len(matches)):
        if (matches[i]):
            max_distance = np.max([match.distance for match in matches[i]])
            print(max_distance)
            matches[i] = [match for match in matches[i] if match.distance < 0.6 * max_distance]
    
    max_distance = np.max([match.distance for match in ssd_matches])
    ssd_matches = [match for match in ssd_matches if match.distance < 0.6 * max_distance]

    for i in range(len(images)):
        out_img = cv2.drawMatches(base, control_features, images[i], test_features[i], matches[i], None)
        cv2.imwrite("base_"+names[i][:-4]+".jpg", out_img)

    out_img = cv2.drawMatches(base, control_features, images[0], test_features[0], ssd_matches, None)
    cv2.imwrite("base_"+names[0][:-4]+"_SSDRATIO.jpg", out_img)

    # print(threshold_count)
    # print(responses[0:50])
    t = plt.figure(1)
    output = base.copy()

    # Set the circle size and color
    circle_size = 10
    circle_color = (0, 255, 0)

    for point in control_features:
        coord = point.pt
        x, y = int(coord[1]), int(coord[0])
        cv2.circle(output, (y, x), circle_size, circle_color, -1)
    cv2.imwrite("base.jpg", output)
    plt.imshow(output)

    threshold_values = np.linspace(0, 1, 5000) 

    num_interest_points = []
    max_response = np.max(responses)
    for threshold in threshold_values:
        threshold_response = threshold * max_response
        num_points = np.sum(responses > threshold_response)
        num_interest_points.append(num_points)

    plt.figure()
    plt.plot(threshold_values, num_interest_points)
    plt.xlabel('Threshold Value')
    plt.ylabel('Number of Interest Points')
    plt.yscale("log")
    plt.title('Interest Points vs Threshold Value')
    plt.grid(True)
    plt.savefig("actually.jpg")
    plt.show()

    plt.show()

if __name__ == "__main__":
    main()


