import numpy as np

matrix1 = [[1,2,3],[1,2,3]]
matrix2 = np.array([[8,2,3],[4,1,6]])
gXX = 255*255
gYY = 255*255
gXY = 255
response = (gXX * gYY) - (gXY)**2 - 0.05*((gXX+gYY)**2)

des1 = np.array([1,2,3,4,5,6,7,8])
des2 = np.array([2,3,4,5,6,7,8,9])

array = [3, 7, 2, 5, 1.9]
min = np.argpartition(array, 2)[0:2]
ratio = array[min[0]]/array[min[1]]

print(ratio)