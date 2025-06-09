import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

url = "https://services7.arcgis.com/1EmryaM5E3wkdnU/arcgis/rest/services/Ethnicity_Felix_2001/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson"
felix_lang = gpd.read_file(url)

felix_lang = felix_lang[felix_lang["FAMILY"].notna()]

encoder = LabelEncoder()
felix_lang["family_encoded"] = encoder.fit_transform(felix_lang["FAMILY"])

kmeans = KMeans(n_clusters=6, random_state=42)
felix_lang["language_cluster"] = kmeans.fit_predict(felix_lang[["family_encoded"]])

fig, ax = plt.subplots(figsize=(16, 16))
felix_lang.plot(column="language_cluster", cmap="tab20", legend=True, edgecolor="black", linewidth=0.3, ax=ax)
ax.set_title("Language-Based Clusters in Africa (Felix 2001)", fontsize=18)
ax.axis("off")
plt.tight_layout()
plt.show()
