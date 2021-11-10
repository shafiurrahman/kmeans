import pandas as pd
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
#  # this is for jupyter display
cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()

#!pip install pandas ---in terminal

cust_df.columns
df=cust_df.drop("Address",axis=1)
df.head()

from sklearn.preprocessing import StandardScaler
X=df.values[:,1:]
X=np.nan_to_num(X)
clust_dataset=StandardScaler().fit_transform(X)
clust_dataset

clusterNum=3
k_means=KMeans(init="k-means++",n_clusters=clusterNum,n_init=12)
k_means.fit(X)
labels=k_means.labels_
print(labels)

df["Clus_km"]=labels
df.head()

#We can easily check the centroid values by averaging the features in each cluster
df.groupby('Clus_km').mean()

#Now, lets look at the distribution of customers based on their age and income:
area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()


#3d plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(float))

