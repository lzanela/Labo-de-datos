import numpy as np
import pandas as pd
import random
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

df=pd.read_csv("datos_clase_clustering.csv")
df["ppm"]=df["price"]/df["surface_covered"]

sns.scatterplot(data=df, x= "lon", y="lat", hue="ppm")
plt.show()
plt.close()

X=df[["ppm"]]
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

df["ppm_class"] = kmeans.predict(X)

sns.scatterplot(data=df, x= "lon", y="lat", hue="ppm_class")
plt.show()
plt.close()

X=df[["lon","lat"]]
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X)

df["ubi_class"] = kmeans.predict(X)

sns.scatterplot(data=df, x= "lon", y="lat", hue="ubi_class")
plt.show()
plt.close()

X=df[["surface_total","price"]]
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

df["other_class"] = kmeans.predict(X)

sns.scatterplot(data=df, x= "lon", y="lat", hue="other_class")
plt.show()
plt.close()


k_values = [2, 3, 4, 5, 6, 8, 10, 15]
attributes = [["lon", "lat"], ["ppm"]]

for attrs in attributes:
    for k in k_values:
        X=df[attrs]
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto",algorithm="elkan").fit(X)
        new_attr = "_".join(attrs) + "_class"
        df[new_attr] = kmeans.predict(X)
        sns.scatterplot(data=df, x="lon", y="lat", hue=new_attr).set_title(f"K={k}")
        plt.show()
        plt.close()

e_values = [0.005,0.0075]
for attrs in attributes:
    for e in e_values:
        X=df[attrs]
        clustering = DBSCAN(eps=e, min_samples=5).fit(X)
        new_attr = "_".join(attrs) + "_class"
        df[new_attr] = clustering.fit_predict(X)
        sns.scatterplot(data=df, x="lon", y="lat", hue=new_attr).set_title(f"E={e}")
        plt.show()
        plt.close()