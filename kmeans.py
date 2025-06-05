import numpy as np
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
plt.subplots(2,2)
img = cv2.imread('LDN00109.jpg')
plt.subplot(2,2,1)
plt.title("ảnh gốc")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
K = 3
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(X)
labels = kmeans.predict(X)
img4 = np.zeros_like(X)
for i in range(K):
    img4[labels == i] = kmeans.cluster_centers_[i]
plt.subplot(2,2,2)
plt.title("sau khi dùng kmeans với K = {}".format(K))
img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
plt.show()
