import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2
from scipy.spatial.distance import cdist

img = cv2.imread('coin.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img)

X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

def Kmean_from_sratch(X, K:int):
    def kmeans_init_centers(X, k):
        return X[np.random.choice(X.shape[0], k, replace=False)]
    
    def kmeans_assign_labels(X, centers):
        D = cdist(X, centers)
        return np.argmin(D, axis = 1)
    
    def kmeans_update_centers(X, labels, K):
        centers = np.zeros((K,X.shape[1]))
        for k in range(K):
            Xk = X[labels == k, :]
            centers[k,:] = np.mean(Xk, axis = 0)
        return centers
    
    def has_converged(centers, new_centers):
        return (set([tuple(a)]))
    