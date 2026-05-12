import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

drugs = pd.read_csv("primary-screen-replicate-collapsed-treatment-info.csv")
logfc = pd.read_csv("primary-screen-replicate-collapsed-logfold-change.csv")
cells = pd.read_csv("primary-screen-cell-line-info.csv")
#PCA first 
logfc.head()
# assume first column is drug identifier
logfc = logfc.set_index(logfc.columns[0])
X = logfc.select_dtypes(include=["number"])
X = X.fillna(X.mean())
#standardize data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=50)  # start with 20–50
X_pca = pca.fit_transform(X_scaled)

explained = np.cumsum(pca.explained_variance_ratio_)
print(explained[:10])


plt.plot(explained)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Explained Variance")
plt.show()

pca_2 = PCA(n_components=2)
X_2d = pca_2.fit_transform(X_scaled)

plt.scatter(X_2d[:,0], X_2d[:,1], s=5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Drug Response Landscape (PCA)")
plt.show()

loadings = pca_2.components_.T  # shape: (features x PCs)

loading_df = pd.DataFrame(
    loadings,
    columns=["PC1", "PC2"],
    index=X.columns
)

plt.figure(figsize=(7,6))

plt.scatter(loading_df["PC1"], loading_df["PC2"], s=10)

for i, txt in enumerate(loading_df.index[:30]):  # label only first 30 to avoid clutter
    plt.annotate(txt, (loading_df["PC1"][i], loading_df["PC2"][i]), fontsize=6)

plt.xlabel("PC1 loading")
plt.ylabel("PC2 loading")
plt.title("PCA Loadings Plot (Features / Cell Lines)")
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.show()

