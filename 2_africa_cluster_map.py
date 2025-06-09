import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import unicodedata

original_df = pd.read_csv("Prep for Clustering.csv")
clustered_df = pd.read_csv("Clustered_Africa_Results.csv")
clustered_df['country'] = original_df['country'].astype(str).str.strip()
clustered_df[['country', 'cluster']].to_csv("Map_Country_Clusters.csv", index=False)
clusters = pd.read_csv("Map_Country_Clusters.csv")

def clean_text(s):
    if pd.isna(s): return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("utf-8")
    s = s.strip().lower()
    s = s.replace('\xa0', ' ').replace('\u200b', '')
    s = s.replace("*", "").replace("”", "").replace("“", "")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s

clusters['country_clean'] = clusters['country'].apply(clean_text)

name_map = {
    "ivory coast": "cote d'ivoire",
    "swaziland": "eswatini",
    "cape verde": "cabo verde",
    "la reunion": "reunion",
    "dr congo": "congo, democratic republic",
    "congo": "congo, republic",
    "gambia": "the gambia",
    "western sahara": "western sahara"
}
reverse_map = {v: k for k, v in name_map.items()}
clusters['country_clean'] = clusters['country_clean'].replace(reverse_map)

africa_shapes = gpd.read_file("africa.geojson")
africa_shapes['name_clean'] = africa_shapes['name'].apply(clean_text)
africa = africa_shapes.merge(clusters, how="left", left_on="name_clean", right_on="country_clean")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
africa.plot(
    column='cluster',
    cmap='Set1',
    linewidth=0.5,
    edgecolor='black',
    legend=True,
    ax=ax
)

plt.title("Redrawn African Regions Based on Machine Learning Clusters", fontsize=14)
plt.axis('off')
plt.show()

print("\nTop 20 countries:")
print(africa[['name', 'country', 'cluster']].head(20))

missing = africa[africa['cluster'].isna()]
print(f"\nMissing clusters: {missing.shape[0]}")
print("Unmatched countries:\n", missing['name'].sort_values().tolist())
