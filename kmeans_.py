# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:42:40 2019

@author: alari
"""
from numpy.linalg import norm
import numpy as np
class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def initialise_centroids(self, X):
        """
        Choose first n_clusters data points as cluster centroids
        """
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        """
        Update n cluster centroids, that have X.shape[1] independent coordinates
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters)) # initialize distance matrix, each data point from each cluster
        for k in range(self.n_clusters): # for each cluster centroid
            row_norm = norm(X - centroids[k, :], axis=1) # find distance of data points to that centroid
            distance[:, k] = np.square(row_norm) # save the squared distances to the distance matrix
        return distance # return the distance matrix

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1) # find the index j that minimises the distance

    def compute_sse(self, X, labels, centroids):
        """
        computes sum of squared errors of each data point from the centroids
        """
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance)) 
    
    def fit(self, X):
        """
        The fit method fits the K-Means algorithm to max iterations
        """
        self.centroids = self.initialise_centroids(X) # first we randomly initialise cluster centroids
        for i in range(self.max_iter):
            old_centroids = self.centroids # save the previous iteration centroids
            distance = self.compute_distance(X, old_centroids) # compute distances from previous iteration centroids
            self.labels = self.find_closest_cluster(distance) # label the data points based on distance to closest centroid
            self.centroids = self.compute_centroids(X, self.labels) # update the labels of the data points
            if np.all(old_centroids == self.centroids): # if no change in the centroids
                break                                  #
        self.error = self.compute_sse(X, self.labels, self.centroids)
    
    def predict(self, X):
        distance = self.compute_distance(X, self.centroids) # predict the 
        return self.find_closest_cluster(distance)
