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

def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    print(dim)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def invert_rb(img):
    img0 = np.copy(img[:,:,0])
    img2 = np.copy(img[:,:,2])
    img[:,:,0], img[:,:,2] = img2, img0
    return img

def run_clustering(img, nbclust=3):
    
    kmeans = KMeans(12).fit(img)
    clusteredimg = np.zeros((lowdim[0], lowdim[1], 3))
    clusterlabels = np.reshape(kmeans.labels_, newshape=(lowdim[0], lowdim[1]))
    
    centroids = kmeans.cluster_centers_
    
    for i in range(lowdim[0]):
        for j in range(lowdim[1]):
            clusteredimg[i,j] = centroids[clusterlabels[i,j],:] 
    return clusteredimg, centroids


def makepadd(img, kernsize=5):
    imgsize = img.shape
    nbsup = kernsize // 2
    ressize = (imgsize[0] + 2*nbsup, imgsize[1] + 2*nbsup)
    res = np.zeros((ressize))
    res[nbsup:-nbsup, nbsup:-nbsup] = img
    for i in range(nbsup):
        res[i,nbsup:-nbsup] = img[0,:]
        res[-i,nbsup:-nbsup] = img[-1,:]
        
        res[nbsup:-nbsup, i] = img[:,0]
        res[nbsup:-nbsup, -i] = img[:,-1]
    return res

def run_morphograd(img, kernsize=5):
    imgsize = img.shape
    assert kernsize%2 == 1
    img = makepadd(img, kernsize)
    
    res = np.zeros((imgsize))
    
    for i in range(imgsize[0]):
        for j in range(imgsize[1]):
            window = img[i:i+kernsize, j:j+kernsize]
            res[i,j] = np.max(window) - np.min(window)
    return res
    

if __name__=='__main__':
    
    seagull = cv2.imread("seagull.jpg", cv2.IMREAD_GRAYSCALE)
    # seagull[:,:,2], seagull[:,:,1] = seagull[:,:,1], seagull[:,:,2]
    
    seagull = rescale_frame(seagull, scale=0.5)
    # seagull = invert_rb(seagull)
    lowdim = seagull.shape
    seagull = run_morphograd(seagull)
    # seagull = np.reshape(seagull, newshape=(lowdim[0]*lowdim[1],-1))
    
    # clusteredseagull, centroids = run_clustering(seagull, nbclust=3)
    plt.imshow(seagull)
    # print(centroids)
    cv2.imwrite("morphoseagull.jpg", seagull)
    print(seagull.shape)
