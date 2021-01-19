# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 15:37:34 2021

@author: DIY
"""

import matplotlib.pyplot as plt 
import numpy as np              
import sklearn.cluster as sklc  
import sklearn.metrics as sklm  


data_filename = 'C:\\Users\\DIY\\Desktop\\qmcw3\\processed data\\new.csv'
num_clusters = 2
figure_width, figure_height = 7,7

data = np.genfromtxt(data_filename,delimiter = ',')


fig_title = 'Figure Title'
x_label   = 'x-axis label'
y_label   = 'y-axis label'
title_fontsize = 15
label_fontsize = 10
x_min, x_max = 0.95*np.min(data[1:,2]), 1.05*np.max(data[1:,2])
y_min, y_max = 0.4*np.min(data[1:,3]), 1.2*np.max(data[1:,3])

def setup_figure():
    
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel(x_label,fontsize=label_fontsize)
    plt.ylabel(y_label,fontsize=label_fontsize)
    
x_values = data[1:,2]
y_values = data[1:,3]

#And then a cheeky plot:
plt.figure(0,figsize=(figure_width,figure_height))
setup_figure()
plt.xlabel(x_label,fontsize=label_fontsize)
plt.ylabel(y_label,fontsize=label_fontsize)
plt.title(fig_title,fontsize=title_fontsize)
plt.plot(x_values,y_values, 'k.')

kmeans_output = sklc.KMeans(n_clusters=num_clusters, n_init=1).fit(data)
print(kmeans_output)

clustering_ids_kmeans = kmeans_output.labels_
print(clustering_ids_kmeans)

complete_data_with_clusters = np.hstack((data,np.array([clustering_ids_kmeans]).T))
print(complete_data_with_clusters)

data_by_cluster = []

for i in range(num_clusters):
    
    this_data = []
    
    for row in complete_data_with_clusters:
        
        if row[-1] == i:
            this_data.append(row)
    
    this_data = np.array(this_data)
    
    data_by_cluster.append(this_data)
data_by_cluster

color_list = ['b','r','g','m','c','k','y']
for i in range(num_clusters):
    
    plt.figure(i+1,figsize=(figure_width,figure_height))
    setup_figure()
    plt.title(fig_title + ' - Cluster ' + str(i),fontsize=title_fontsize)
    
    x_values = data_by_cluster[i][1:,2]
    y_values = data_by_cluster[i][1:,3]
    
    plt.plot(x_values,y_values,color_list[i % num_clusters] + '.')

plt.figure(num_clusters + 1,figsize=(figure_width,figure_height))
setup_figure()
plt.title(fig_title + ' - Coloured by Cluster',fontsize=title_fontsize)

for i in range(num_clusters):
    
    x_values = data_by_cluster[i][1:,2]
    y_values = data_by_cluster[i][1:,3]
    
    plt.plot(x_values,y_values,color_list[i % num_clusters] + '.')


silhouette_kmeans = sklm.silhouette_score(data,clustering_ids_kmeans)
print("Silhouette Score:", silhouette_kmeans)
















