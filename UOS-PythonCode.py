#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:40:04 2019

@author: emanuel
"""
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from pandas import DataFrame
from sklearn import metrics
import statistics
import csv
import math
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
from pykalman import KalmanFilter

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def DBSCAN_Clusterization(X, EPS, MIN_SAMPLES):
    
    DBClusters = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric ='euclidean',algorithm = 'auto')#'kd_tree')
    DBClusters.fit(X)
    #DBClusters.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(DBClusters.labels_)) - (1 if -1 in DBClusters.labels_ else 0)
    core_samples = np.zeros_like(DBClusters.labels_, dtype = bool)
    core_samples[DBClusters.core_sample_indices_] = True
    
    # PRINT CLUSTERS & # of CLUSTERS
#    print("Clusters:"+str(DBClusters.labels_))
#    
#    print('Estimated number of clusters: %d' % n_clusters_)
    
    clusters = [X[DBClusters.labels_ == i] for i in range(n_clusters_)]
    outliers = X[DBClusters.labels_ == -1]
    
    # Plot Outliers
#    plt.scatter(outliers[:,0], outliers[:,1], c="black", label="Outliers")
    
    
    # Plot Clusters
#    cmap = get_cmap(len(clusters))
    x_clusters = [None] * len(clusters)
    y_clusters = [None] * len(clusters)
    #colors = [0]
#    colors = "bgrcmykw"
    color_index = 0
    for i in range(len(clusters)):
        x_clusters[i] = []
        y_clusters[i] = []
       # print("Tamano Cluster "+ str(i) + ": " + str(len(clusters[i])))
        for j in range(len(clusters[i])):
            x_clusters[i].append(clusters[i][j][0])
            y_clusters[i].append(clusters[i][j][1])
            
    #        
        
#        plt.scatter(x_clusters[i], y_clusters[i], label= "Cluster %d" %i,  s=8**2, c=colors[color_index]) #c=cmap(i)) 
        color_index += 1
        
    
    #plot the Clusters 
    #plt.title("Clusters Vs Serving UABS")
#    plt.scatter(x2,y2,c="yellow", label= "UABSs", s=10**2) #plot UABS new position
#    plt.xlabel('x (meters)', fontsize = 16)
#    plt.ylabel('y (meters)', fontsize = 16)
#    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#              fancybox=True, shadow=True, ncol=5)
#    plt.savefig("Graph_Clustered_UOS_Scenario.pdf", format='pdf', dpi=1000)
#    plt.show()      
    
    return clusters, x_clusters, y_clusters  


def Sum_Avg_Parameter(clusters,x3,Metric_Flag):
    #Sum of SINR and mean to later prioritize the clusters  
    SUMSinr = [None] * len(clusters)
    
    for i in range(len(clusters)):
        SUMSinrClusters = 0
        for j in range(len(clusters[i])):
            index_x3 = np.where(x3 == clusters[i][j][0])
    #        print("Found x3: "+str(np.where(x3 == clusters[i][j][0]))) # para comparar con x3
    #        print("Found y3: "+str(np.where(y3 == clusters[i][j][1]))) # para comparar con x3
            for k in range(len(index_x3)):
                    if Metric_Flag == 0:
                        if (y3[index_x3[k]] == clusters[i][j][1]):
    #                print("SINR FOUND: " + str(sinr[index_x3[k]]))
                            SUMSinrClusters += sinr[index_x3[k]]
                    elif Metric_Flag == 1:
                        if (y4[index_x3[k]] == clusters[i][j][1]):
                            SUMSinrClusters += UE_Throughput[index_x3[k]]
                    elif Metric_Flag == 2:
                        if (y4[index_x3[k]] == clusters[i][j][1]):
                            SUMSinrClusters += UE_Delay[index_x3[k]]
                    elif Metric_Flag == 3:
                        if (y4[index_x3[k]] == clusters[i][j][1]):
                            SUMSinrClusters += UE_Packet_Loss[index_x3[k]]
    #                print(sinr[index_x3[k]])
    #                print(SUMSinrClusters)
    #   SUMSinr[i] = sinr[index_x3[k]]
        SUMSinr[i] = SUMSinrClusters        
    
    SINRAvg = [None] * len(clusters)
    
    for i in range(len(SUMSinr)):
        SINRAvg[i] = SUMSinr[i]/len(clusters[i])
        
    return SINRAvg

def Priotirize(SINRAvg):
    #Prioritize by greater SINR    
    CopySINRAvg = SINRAvg.copy()
    SINRAvgPrioritized = []
    for i in range(len(SINRAvg)):
        #print("SINR Max:" + str(max(CopySINRAvg)))
        SINRAvgPrioritized.append(min(CopySINRAvg))  #evaluar si es MAX o MIN que quiero para obtener el cluster con mayor SINR
        CopySINRAvg.remove(min(CopySINRAvg))
    
    return SINRAvgPrioritized

def Centroids_Clusters(clusters,x_clusters,y_clusters):
    #Centroids - median of clusters
    x_clusters_mean = [None] * len(clusters)
    y_clusters_mean = [None] * len(clusters)
    for i in range(len(clusters)):
        x_clusters_mean[i] = []
        y_clusters_mean[i] = []
        x_clusters_mean[i].append(statistics.mean(x_clusters[i]))
        y_clusters_mean[i].append(statistics.mean(y_clusters[i]))
        
    Centroids = list(zip([i[0] for i in x_clusters_mean],[i[0] for i in y_clusters_mean]))
    
    return Centroids

def Reorder_Centroids(Centroids, SINRAvg, SINRAvgPrioritized):
    #Reorder Centroides based on prioritized AVGSINR
    CentroidsPrio = []   
    for i in range(len(SINRAvg)):
        index_SAP = np.where(SINRAvg == SINRAvgPrioritized[i] )
    #    print(index_SAP[0])
    #    print(Centroids[int(index_SAP[0])])
        CentroidsPrio.append(Centroids[int(index_SAP[0])])
        
    #for i in CentroidsPrio:
    #    print("{} {} ".format(i[0], i[1]))
    #centroidsarray = np.asarray(Centroids)
    #print(centroidsarray)
    
    return  CentroidsPrio

# KNN Implementation for finding the nearest UABS to the X Centroid.
# Create the knn model.
def nearest_UABS(UABSCoordinates, cellIds, Centroids):
      Kneighbors = 2
      if len(cellIds) == 1:
        Kneighbors = 1
      knn = KNeighborsClassifier(n_neighbors= Kneighbors, weights= "uniform" ,algorithm="auto")
      knn.fit(UABSCoordinates,cellIds)
      #predict witch UABS will be serving to the X Centroid.
      Knnpredict= knn.predict(Centroids)
      return Knnpredict

#-----------------------------------Main---------------------------------------------------------------- 

#-----------------------------------Import data files---------------------------------------------------#
with open('enBs') as fenBs:
    data1 = np.array(list((float(x), float(y), float(z), int(cellid)) for x, y, z, cellid in csv.reader(fenBs, delimiter= ',')))
    
with open('LTEUEs') as fUEs:
    data2 = np.array(list((float(x), float(y), float(z)) for x, y, z in csv.reader(fUEs, delimiter= ',')))
    
with open('UABSs') as fUABS:
    data3 = np.array(list((float(x), float(y), float(z), int(cellid)) for x, y, z, cellid in csv.reader(fUABS, delimiter= ',')))

with open('UEsLowSinr') as fUEsLow:
    data4 = np.array(list((float(x), float(y), float(z), float (Sinr), int (Imsi),int(cellid)) for x, y, z, Sinr,Imsi, cellid in csv.reader(fUEsLow, delimiter= ',')))

#with open('UABS_Energy_Status') as fUABS_Energy:
#    data5 = np.array(list((int(time), int(UABSID), int(Remaining_Energy)) for time, UABSID, Remaining_Energy in csv.reader(fUABS_Energy, delimiter= ',')))

with open('UEs_UDP_Throughput') as fUE_QoS:
    data6 = np.array(list((int(time), int(UE_ID), float(x), float(y), float(z), float(UE_Throughput), float(UE_Delay) , float(UE_Packet_Loss)) for time, UE_ID, x, y, z, UE_Throughput, UE_Delay, UE_Packet_Loss in csv.reader(fUE_QoS, delimiter= ',')))


#---------------Parse Data----------------------#

#----------enBs--------------#
x,y,z, cellid= data1.T

#----------Total LTE Users--------------#
x1,y1,z1= data2.T

#----------UABS--------------#
x2,y2,z2, cellid3= data3.T
UABSCoordinates = np.array(list(zip(x2,y2)))

#----------Users with Low SINR--------------#
if (data4.size != 0):
    x3,y3,z3, sinr, imsi, cellid4= data4.T
    X = np.array(list(zip(x3,y3)))

#----------UABS Energy--------------#
#if (data5.size != 0):
#    time, Uabs_Id, Remaining_Energy = data5.T

#----------QoS Parameters--------------#
if (data6.size != 0):
    # Normalize throughput, delay and Packet Loss columns
     data6[:,5] = preprocessing.normalize([data6[:,5]])
     data6[:,6] = preprocessing.normalize([data6[:,6]])
     data6[:,7] = preprocessing.normalize([data6[:,7]])
     time_UE, UE_ID, x4, y4, z4, UE_Throughput, UE_Delay, UE_Packet_Loss = data6.T
## ----------------Here i have to just create a X Y pair with lowest throughput users.
     X1 = np.array(list(zip(x4,y4)))
    


#---------------Clustering with DBSCAN for Users with Low SINR---------------------
eps_low_SINR=600
min_samples_low_SINR=2
if (data4.size != 0):
    clusters, x_clusters, y_clusters = DBSCAN_Clusterization(X, eps_low_SINR, min_samples_low_SINR)


#---------------Clustering with DBSCAN for Users with Low Throughput---------------------
eps_low_tp=600
min_samples_low_tp=2
if (data6.size != 0):
    clusters_QoS, x_clusters_QoS, y_clusters_QoS = DBSCAN_Clusterization(X1, eps_low_tp, min_samples_low_tp)
 

 #Sum of SINR and mean to later prioritize the clusters
SINRAvg = []
Metric_Flag = 0
if (data4.size != 0):
    SINRAvg= Sum_Avg_Parameter(clusters,x3, Metric_Flag)
weight_SINR_Total = 0.0
weight_SINR = 0.6 

#Sum of Throughput and mean to later prioritize the clusters
Metric_Flag = 1
if (data6.size != 0): 
    QoS_Throughput_Avg= Sum_Avg_Parameter(clusters_QoS,x4, Metric_Flag)
    #print("Stdev: "+ str(np.std(QoS_Throughput_Avg)))
    
#Sum of Delay and mean to later prioritize the clusters
Metric_Flag = 2
if (data6.size != 0): 
    QoS_Delay_Avg= Sum_Avg_Parameter(clusters_QoS,x4, Metric_Flag)
    
#Sum of Packet Loss and mean to later prioritize the clusters
Metric_Flag = 3
if (data6.size != 0): 
    QoS_PLR_Avg= Sum_Avg_Parameter(clusters_QoS,x4, Metric_Flag)

#Calculate total weight of QoS Clustering
if (data6.size != 0):
    weight_QoS_Throughput = 0.4
    weight_QoS_Delay = 0.3
    weight_QoS_PLR = 0.3
    weight_QoS_Total = 0.0
    weight_QoS = 0.4

#Weight of Throughput + Weight of Delay + Weight of PLR = 1

    for i in range(len(QoS_Throughput_Avg)):
        weight_QoS_Total += ((QoS_Throughput_Avg[i]*weight_QoS_Throughput)+(QoS_Delay_Avg[i]*weight_QoS_Delay)+(QoS_PLR_Avg[i]*weight_QoS_PLR))
#        print("QoS: "+ str((QoS_Throughput_Avg[i]*weight_QoS_Throughput)+(QoS_Delay_Avg[i]*weight_QoS_Delay)+(QoS_PLR_Avg[i]*weight_QoS_PLR)))
#    SINRAvg_norm = SINRAvg.copy()   
    for sinr in SINRAvg:
#        SINRAvg_norm[i] = preprocessing.normalize(SINRAvg[i])
        weight_SINR_Total += sinr*weight_SINR
        
    #if (len(weight_SINR_Total) > len(weight_QoS_Total)):
    if (weight_SINR_Total > weight_QoS_Total):
    #    print("There are more SIRN clusters than QoS Clusters: " + str(len(weight_SINR_Total)) + " vs " + str(len(weight_QoS_Total)))
    #    print("UABS will be positionated by Low SINR")
        #Prioritize by greater SINR or QoS
        SINRAvgPrioritized = Priotirize(SINRAvg) #Here we reorder-prioritize based on the clusters with min SINR.
         
        ##Convert SINR to dB just to see which cluster has bigger SINR    
        #SINRinDB = []
        #for i in range(len(SINRAvgPrioritized)):
        #      SINRinDB.append(10 * math.log(SINRAvgPrioritized[i]))  
               
        #Centroids - median of clusters
        Centroids = Centroids_Clusters(clusters,x_clusters,y_clusters)
        
        #Reorder Centroides based on prioritized AVGSINR
        CentroidsPrio = Reorder_Centroids(Centroids, SINRAvg, SINRAvgPrioritized)
        
        file = open("UOS_Clustering_Decitions.txt","a") 
     
        file.write("0,SINR clusters have more weight than QoS Clusters: " + str(weight_SINR_Total) + " vs " + str(weight_QoS_Total) + "\n") 
        file.write("UABS will be positionated by Low SINR" + "\n")
         
        file.close() 
        
    else:
    #    print("There are more QoS clusters than SINR Clusters: " + str(len(weight_QoS_Total)) + " vs " + str(len(weight_SINR_Total)))
    #    print("UABS will be positionated by Low QoS")
        #Prioritize by greater SINR or QoS
        QoSAvgPrioritized = Priotirize(QoS_Throughput_Avg)  #Here we reorder-prioritize based on the clusters with min throughput.
               
        #Centroids - median of clusters
        Centroids = Centroids_Clusters(clusters_QoS,x_clusters_QoS,y_clusters_QoS)
        
        #Reorder Centroides based on prioritized AVG_QoS
        CentroidsPrio = Reorder_Centroids(Centroids, QoS_Throughput_Avg, QoSAvgPrioritized)
        
        file = open("UOS_Clustering_Decitions.txt","a") 
     
        file.write("1,QoS clusters have more weight than SINR Clusters: " + str(weight_QoS_Total) + " vs " + str(weight_SINR_Total) + "\n") 
        file.write("UABS will be positionated by Low QoS" + "\n")
         
        file.close() 
        
if (data6.size == 0):    
    #Prioritize by greater SINR or QoS
    SINRAvgPrioritized = Priotirize(SINRAvg)
        
        
    ##Convert SINR to dB just to see which cluster has bigger SINR    
    #SINRinDB = []
    #for i in range(len(SINRAvgPrioritized)):
    #      SINRinDB.append(10 * math.log(SINRAvgPrioritized[i]))  
               
        
    #Centroids - median of clusters
    Centroids = Centroids_Clusters(clusters,x_clusters,y_clusters)
        
        
    #Reorder Centroides based on prioritized AVGSINR
    CentroidsPrio = Reorder_Centroids(Centroids, SINRAvg, SINRAvgPrioritized)

if  (CentroidsPrio):
    while len(CentroidsPrio) > 0 and len(cellid3) > 0:
        used_UABS_ids = set()
        nearest = nearest_UABS(UABSCoordinates, cellid3, CentroidsPrio)
        j=0
        for i in CentroidsPrio:
            if nearest[j] in used_UABS_ids: break
            print("{} {} {} ".format(i[0], i[1], nearest[j]))
            used_UABS_ids.add(nearest[j])
            j+=1
        #find indices of UABSs coordinates to delete
        indices = [i for i in range(len(cellid3)) if cellid3[i] in used_UABS_ids]
        #delete coordinates by index
        UABSCoordinates = np.delete(UABSCoordinates, indices, 0)
        cellid3 = [x for x in cellid3 if x not in used_UABS_ids]
        CentroidsPrio = CentroidsPrio[j:]
else:
      for i in CentroidsPrio:
            print("{} {} ".format(i[0], i[1]))

#scores = {}
#scores_list = []
#for k in range(Kneighbors):
#    scores[k] = metrics.accuracy_score(cellid3,Knnpredict)
#    scores_list.append(metrics.accuracy_score(cellid3,Knnpredict))
