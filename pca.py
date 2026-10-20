# LAB EXERCISE 6 - PRINCIPAL COMPONENT ANALYSIS (PCA)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.DataFrame({
    "Maths": [80, 75, 90, 60, 85, 70],
    "Science": [85, 78, 95, 65, 88, 72],
    "English": [70, 82, 88, 75, 80, 68],
    "Computer": [90, 85, 98, 70, 92, 78]
})
print("Original data:\n", df)

scaled = StandardScaler().fit_transform(df)
pca = PCA(n_components=2)
components = pca.fit_transform(scaled)

print("\nPrincipal component scores:\n", pd.DataFrame(
    components, columns=["PC1", "PC2"]
))
print("\nExplained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", pca.explained_variance_ratio_.sum())
print("\nComponent loadings:\n", pd.DataFrame(
    pca.components_.T, index=df.columns, columns=["PC1", "PC2"]
))
