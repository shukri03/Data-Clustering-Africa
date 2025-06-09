from pathlib import Path 
import pandas as pd, textwrap, sys

ROOT = Path.cwd()
CSV  = ROOT / "Map_Country_Clusters.csv"
PREP = ROOT / "Prep for Clustering - Sheet1 (2).csv" 

need_cols = {"country_clean", "cluster"}
df = pd.read_csv(CSV).rename(columns=str.lower)     
missing_cols = need_cols - set(df.columns)
if missing_cols:
    sys.exit(f"{CSV.name} is missing column(s): {', '.join(missing_cols)}")

try:
    df["cluster"] = pd.to_numeric(df["cluster"])
except ValueError as e:
    sys.exit("“cluster” column must contain numbers only.")

good_names = pd.read_csv(PREP, usecols=["country"])["country"].str.strip()
if (df["country_clean"].str.match(r"^\d+(\.\d+)?$")).any():
    if len(good_names) != len(df):
        sys.exit("Row counts differ between PREP file and CLUSTERS file.")
    df["country_clean"] = good_names.values
    df.to_csv(CSV, index=False)
    print("Fixed numeric placeholders with real country names.")

nan_list = df.loc[df["cluster"].isna(), "country_clean"].tolist()
if nan_list:
    txt = textwrap.fill(", ".join(nan_list), 80)
    sys.exit(f"Add a cluster number for these countries then rerun:\n{txt}")

print("all good — proceed to Cell 2")

from pathlib import Path
import geopandas as gpd, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path.cwd()
SHP  = ROOT / "Murdock_shapefile_v2.shp"
CSV  = ROOT / "Map_Country_Clusters.csv"

clean = lambda s: str(s).lower().strip()

gdf = (gpd.read_file(SHP).to_crs("EPSG:4326")
         .query("CONTINENT=='Africa'")
         .assign(key=lambda d: d["ADMIN"].map(clean)))

df  = (pd.read_csv(CSV)[["country_clean", "cluster"]]
         .rename(columns={"country_clean": "country"})
         .assign(key=lambda d: d["country"].map(clean)))

gdf = gdf.merge(df, on="key", how="left", validate="m:1")
print(f"{gdf['cluster'].isna().sum()} small polygons still grey (no cluster label)")

cluster_labels = {
    0: "Southern Corridor",
    1: "Sahara Belt",
    2: "Atlantic–Horn Pact",
    3: "Trans-Sahel Collective",
}

cluster_colours = {
    0: "#1f77b4",
    1: "#9ecae9",
    2: "#ff7f0e",
    3: "#ffbb78",
}

gdf_ok = gdf.dropna(subset=["cluster"])

gdf_ok["cluster_name"] = gdf_ok["cluster"].map(cluster_labels)
gdf_ok["colour"]       = gdf_ok["cluster"].map(cluster_colours)

gdf_new = gdf_ok.dissolve(by="cluster_name", as_index=False)

colour_list = [cluster_colours[c] for c in gdf_new["cluster"].unique()]
cmap = ListedColormap(colour_list)

fig, ax = plt.subplots(figsize=(11, 13))
gdf_new.plot(
    column="cluster_name",
    cmap=cmap,
    edgecolor="black",
    linewidth=0.4,
    legend=True,
    categorical=True,
    legend_kwds={"title": "New Country"},
    ax=ax,
)

ax.set_title("Africa Re-imagined — Borders from Data-Driven Clusters",
             fontsize=16, pad=15)
ax.axis("off")
plt.tight_layout()
plt.show()
