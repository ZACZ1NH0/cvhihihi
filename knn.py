import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

def calcHist(image, bins=64, ranges=[0, 256]):
    hist = np.zeros((bins**3), dtype=np.float32)
    pixel_count = 0
    b, g, r = cv2.split(image)
    for channel in (b, r, g):
        channel_hist, _ = np.histogram(channel, bins=bins, range=ranges)
        hist[pixel_count: pixel_count + bins] = channel_hist
        pixel_count +=bins
    return hist

test_img = cv2.imread('LDN00109.jpg')
test_img = cv2.resize(test_img, (300,300))

data_imgs = []
data_hists =[]
for i in range(1,12):
    data_img = cv2.imread('')
    data_img = cv2.resize(data_img, (300, 300))
    hist = calcHist(data_img).flatten()
    data_imgs.append(data_img)
    data_hists.append(hist)

for i in range(1,6):
    data_img = cv2.imread('')
    data_img = cv2.resize(data_img, (300, 300))
    hist = calcHist(data_img).flatten()
    data_imgs.append(data_img)
    data_hists.append(hist)

test_hist = calcHist(test_img).flatten()

K = 3
neigh = NearestNeighbors(n_neighbors=K)
neigh.fit(data_hists)
distances, indices = neigh.kneighbors([test_hist])

cv2.imshow('Test image', test_img)
for i in range(K):
    data_img = data_imgs[indices[0][i]]
    cv2.imshow('Data image {}'.format(i+1),data_img)
cv2.waitKey(0)
cv2.destroyAllWindows()