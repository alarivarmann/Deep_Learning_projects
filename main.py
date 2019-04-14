# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:55:17 2019

@author: DESKTOP
"""
import argparse
import sys
sys.path.append('C:/demo/')
from kmeans_ import Kmeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
#from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


###################################################################################################
def read_normalize_and_fit_visualize(path='C:/demo/data/OldFaithful.csv',**kwargs):
    df = pd.read_csv(path)
    # Standardize the data
    X_norm = StandardScaler().fit_transform(df) # now our data has 0 mean and unit-variance in both variables
    
    # Run local implementation of kmeans
    km = Kmeans(n_clusters=2, max_iter=100)
    km.fit(X_norm)
    centroids = km.centroids
    ######################### PLOTTING PART #########################
    
    fig, ax = plt.subplots(1,2)
    ax = np.ravel(ax)
    
    ax[0].scatter(df.iloc[:, 0], df.iloc[:, 1])
    ax[0].set_xlabel('Eruption time in mins')
    ax[0].set_ylabel('Waiting time to next eruption')
    ax[0].set_title('Visualization of raw data');

    ax[1].scatter(X_norm[km.labels == 0, 0], X_norm[km.labels == 0, 1],
                c='green', label='cluster 1')
    ax[1].scatter(X_norm[km.labels == 1, 0], X_norm[km.labels == 1, 1],
                c='blue', label='cluster 2')
    ax[1].scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,
                c='r', label='centroid')
    ax[1].legend()
    ax[1].set_xlim([-2, 2])
    ax[1].set_ylim([-2, 2])
    ax[1].set_xlabel('Eruption time in mins')
    ax[1].set_ylabel('Waiting time to next eruption')
    ax[1].set_title('Visualization of clustered data', fontweight='bold')
    ax[1].set_aspect('equal')
    plt.show()
    output_dict = {}
    output_dict['kmeans'] = km
    output_dict['norm_data'] = X_norm
    return output_dict
def plot_first_9_iterations():
    X_norm = output_dict['norm_data']
    n_iter = 9
    fig, ax = plt.subplots(3, 3, figsize=(10,10))
    ax = np.ravel(ax)
    centers = []
    for i in range(n_iter):
        # Run local implementation of kmeans
        km = Kmeans(n_clusters=3,
                    max_iter=3,
                    random_state=np.random.randint(0, 1000, size=1))
        km.fit(X_norm)
        centroids = km.centroids
        centers.append(centroids)
        ax[i].scatter(X_norm[km.labels == 0, 0], X_norm[km.labels == 0, 1],
                      c='cyan', label='cluster 1')
        ax[i].scatter(X_norm[km.labels == 1, 0], X_norm[km.labels == 1, 1],
                      c='magenta', label='cluster 2')
        ax[i].scatter(X_norm[km.labels == 2, 0], X_norm[km.labels == 2, 1],
                      c='yellow', label='cluster 3')
        ax[i].scatter(centroids[:, 0], centroids[:, 1],
                      c='k', marker='*', s=300, label='centroid')
        ax[i].set_xlim([-2, 2])
        ax[i].set_ylim([-2, 2])
        ax[i].legend(loc='lower right')
        ax[i].set_title(f'KMeans Cumulative RMSE {km.error:.4f}')
        ax[i].set_aspect('equal')
        plt.tight_layout();
    
    plt.show()
def demonstrate_elbow_method(output_dict):
    X_norm = output_dict['norm_data']
    sse = []
    list_k = list(range(1, 10))
    
    for k in list_k:
        km = Kmeans(n_clusters=k)
        km.fit(X_norm)
        sse.append(km.error)
    
    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.title("Elbow Method", loc ="center")
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.show()

if __name__ == '__main__':
    """
    To pass your custom data folder, use the --path argument, specify the data folder manually
    """
    parser = argparse.ArgumentParser(description='Add K-Means parameters')
    parser.add_argument('--path', type=str,help='string specifying the package folder')
    args = parser.parse_args()
    if args.path is None:
        path = 'C:/demo/data/OldFaithful.csv'
    else :
        path = f"{args.path}/OldFaithful.csv"
    output_dict = read_normalize_and_fit_visualize(path=path)
    ############# lets run 10 ITERATIONS OF KMEANS TO SEE WHAT IS GOING ON ##########
    plot_first_9_iterations()
   
    ################################## ELBOW METHOD #################################
    demonstrate_elbow_method(output_dict = output_dict)

