# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:34:54 2018
Seagull
@author: 16Aghnar
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

seagull = cv2.imread("seagull.jpg")
# seagull[:,:,2], seagull[:,:,1] = seagull[:,:,1], seagull[:,:,2]
 
'''
plt.figure()
plt.subplots(3,1)
plt.subplot(3,1,1)
plt.imshow(seagull[:,:,0])
plt.subplot(3,1,2)
plt.imshow(seagull[:,:,1])
plt.subplot(3,1,3)
plt.imshow(seagull[:,:,2])
'''
sdim = seagull.shape
print(sdim)
lowdim = (sdim[0]//2, sdim[1]//2, 3)
print(lowdim)
def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def invert_rb(img):
    img0 = np.copy(img[:,:,0])
    img2 = np.copy(img[:,:,2])
    img[:,:,0], img[:,:,2] = img2, img0
    return img

seagull = rescale_frame(seagull)
seagull = invert_rb(seagull)
seagull = np.reshape(seagull, newshape=(lowdim[0]*lowdim[1],-1))
kmeans = KMeans(5).fit(seagull)
clusteredseagull = np.zeros((lowdim[0], lowdim[1], 3))
clusterlabels = np.reshape(kmeans.labels_, newshape=(lowdim[0], lowdim[1]))

centroids = kmeans.cluster_centers_

for i in range(lowdim[0]):
    for j in range(lowdim[1]):
        clusteredseagull[i,j] = centroids[clusterlabels[i,j],:] 

plt.imshow(clusteredseagull)
print(centroids)
cv2.imwrite("kmeansseagull_2.jpg", clusteredseagull)
print(seagull.shape)
