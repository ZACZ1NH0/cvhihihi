import cv2
import numpy as np
import matplotlib.pyplot as plt

def cotrang(gray):
    res = np.arange((gray.shape[0])*(gray.shape[1]-1)).reshape((gray.shape[0], gray.shape[1]-1)).astype('uint8')
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1] -1 ):
            if gray[i][j] == 0 or gray[i][j+1] == 0:
                res[i][j] = 0
            else:
                res[i][j] = 255
    return res

def giantrang(gray):
    res = np.arange((gray.shape[0])*(gray.shape[1]-1)).reshape((gray.shape[0], gray.shape[1]-1)).astype('uint8')
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1] -1 ):
            if gray[i][j] == 255 or gray[i][j+1] == 255:
                res[i][j] = 255
            else:
                res[i][j] = 0
    return res

img = cv2.imread('anhdentrang.jpg', cv2.IMREAD_GRAYSCALE)
img_co = cotrang(img)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Kernel 3x3
kernel1 = np.ones((3, 3), np.uint8)
kernel2 = np.ones((2,2), np.uint8)

# Co ảnh (Erosion)
erosion = cv2.erode(binary, kernel1, iterations=1)
dilated = cv2.dilate(binary, kernel1, iterations=1)
_, binary1 = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
final = cv2.erode(binary1, kernel2, iterations=1)
_, binary2 = cv2.threshold(final, 127, 255, cv2.THRESH_BINARY)
final1 = cv2.dilate(binary2, kernel2, iterations=1)
_, binary3 = cv2.threshold(final1, 127, 255, cv2.THRESH_BINARY)
final2 = cv2.erode(binary3, kernel2, iterations=1)
_, binary4 = cv2.threshold(final2, 127, 255, cv2.THRESH_BINARY)
final3 = cv2.erode(binary4, kernel2, iterations=1)

img_gian = giantrang(img)
plt.subplot(2,2,1)
plt.title('Anh goc')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

plt.subplot(2,2,2)
plt.title('Anh co')
plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB))

plt.subplot(2,2,3)
plt.title('Anh gian')
plt.imshow(cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB))

plt.subplot(2,2,4)
plt.title('Anh gian')
plt.imshow(cv2.cvtColor(final3, cv2.COLOR_GRAY2RGB))
plt.show()