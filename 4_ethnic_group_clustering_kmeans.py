import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ethno = gpd.read_file("Murdock_Map_2020.shp")
ethno_proj = ethno.to_crs(epsg=3857)
ethno_proj["centroid"] = ethno_proj.geometry.centroid
ethno_proj["LAT"] = ethno_proj["centroid"].to_crs(epsg=4326).y
ethno_proj["LON"] = ethno_proj["centroid"].to_crs(epsg=4326).x

X = ethno_proj[["LAT", "LON"]].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
k_range = range(2, 16)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Curve for Ethnic Cluster Optimization")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=6, random_state=42)
ethno_proj["cluster"] = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(14, 14))
ethno_proj.plot(column="cluster", cmap="tab20", legend=True, linewidth=0.1, edgecolor="black", ax=ax)
ax.set_title("Ethnic Clusters of Africa (K-Means, Location-Based)", fontsize=16)
ax.axis("off")
plt.show()

fig, ax = plt.subplots(figsize=(14, 14))
ethno.plot(column="NAME", cmap="tab20", legend=False, linewidth=0.1, edgecolor="black", ax=ax)
ax.set_title("Ethnolinguistic Diversity of Africa", fontsize=16)
ax.axis("off")
plt.show()
