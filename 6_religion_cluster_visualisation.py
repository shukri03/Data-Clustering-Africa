import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
import numpy as np

df = pd.read_csv("religion_joshua_data.csv")
df = df.dropna(subset=["X", "Y"])

df["geometry"] = df.apply(lambda row: Point(row["X"], row["Y"]), axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
gdf_proj = gdf.to_crs(epsg=3857)

coords = np.array([[geom.x, geom.y] for geom in gdf_proj.geometry])

kmeans = KMeans(n_clusters=6, random_state=42)
gdf_proj["cluster"] = kmeans.fit_predict(coords)

vor = Voronoi(coords)

polygons = []
for region_index in vor.point_region:
    vertices = vor.regions[region_index]
    if -1 in vertices or len(vertices) == 0:
        polygons.append(None)
        continue
    poly_points = [vor.vertices[i] for i in vertices]
    polygons.append(Polygon(poly_points))

gdf_proj["voronoi_polygon"] = polygons
gdf_poly = gdf_proj.dropna(subset=["voronoi_polygon"]).copy()
gdf_poly = gpd.GeoDataFrame(gdf_poly, geometry="voronoi_polygon", crs=gdf_proj.crs)

cluster_polygons = gdf_poly.dissolve(by="cluster")
cluster_polygons = cluster_polygons.reset_index()

fig, ax = plt.subplots(figsize=(16, 16))
cluster_polygons.plot(
    column="cluster",
    cmap="tab20",
    legend=True,
    alpha=0.6,
    edgecolor="black",
    linewidth=0.8,
    ax=ax,
)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_title("Religious Clusters in Africa (Voronoi Polygons, K-Means)", fontsize=18)
ax.axis("off")
plt.tight_layout()
plt.show()
