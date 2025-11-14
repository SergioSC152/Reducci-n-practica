
# CLUSTERING NO SUPERVISADO EN EL DATASET BREAST CANCER
# Valores de k analizados: 2, 4, 5


# Librerías
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------
# 1. Carga y preprocesamiento
# ------------------------------------------------
print("Cargando el dataset Breast Cancer...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = maligno, 1 = benigno

# Estandarización (crucial con 30 features de escalas distintas)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=data.feature_names)

# Reducción a 3 componentes para visualización
pca = PCA(n_components=3)
X_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns=['PC1', 'PC2', 'PC3'])

# Varianza explicada
print(f"\nVarianza explicada por las 3 primeras componentes: {pca.explained_variance_ratio_}")
print(f"Varianza total explicada: {pca.explained_variance_ratio_.sum():.3f}\n")

# ------------------------------------------------
# 2. Experimentación: k = 2, 4, 5
# ------------------------------------------------
valores_k = [2, 4, 5]
resultados = {}

print("Evaluando Silhouette Score para k = 2, 4, 5...\n")
print("k | Silhouette (datos originales) | Silhouette (datos PCA)")
print("-" * 60)

for k in valores_k:
    # K-Means en datos originales estandarizados
    kmeans_orig = KMeans(n_clusters=k, random_state=42)
    clusters_orig = kmeans_orig.fit_predict(X_scaled)
    sil_orig = silhouette_score(X_scaled, clusters_orig)
    
    # K-Means en datos PCA
    kmeans_pca = KMeans(n_clusters=k, random_state=42)
    clusters_pca = kmeans_pca.fit_predict(X_pca)
    sil_pca = silhouette_score(X_pca, clusters_pca)
    
    resultados[k] = {'original': sil_orig, 'pca': sil_pca}
    print(f"{k} | {sil_orig:28.3f} | {sil_pca:22.3f}")

# ------------------------------------------------
# 3. Conclusiones
# ------------------------------------------------
print("\n" + "="*50)
print("CONCLUSIONES")
print("="*50)

mejor_k_orig = max(resultados, key=lambda k: resultados[k]['original'])
mejor_k_pca = max(resultados, key=lambda k: resultados[k]['pca'])

print(f"\n• Mejor k (datos originales):  k = {mejor_k_orig} → Score = {resultados[mejor_k_orig]['original']:.3f}")
print(f"• Mejor k (datos PCA):         k = {mejor_k_pca} → Score = {resultados[mejor_k_pca]['pca']:.3f}")

print("\n• Interpretación:")
print("  - El dataset tiene DOS clases reales bien separadas (benigno vs maligno).")
print("  - Por lo tanto, k = 2 es la partición natural de los datos.")
print("  - Esperamos que k = 2 tenga el Silhouette Score MÁS ALTO.")
print("  - k = 4 o k = 5 generarán clusters innecesarios y disminuirán la cohesión.")

# ------------------------------------------------
# 4. Visualización 3D para k = 2 (el más significativo)
# ------------------------------------------------
kmeans_viz = KMeans(n_clusters=2, random_state=42)
clusters_viz = kmeans_viz.fit_predict(X_pca)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca['PC1'], X_pca['PC2'], X_pca['PC3'],
                     c=clusters_viz, cmap='coolwarm', s=60, alpha=0.8)
ax.set_title('K-Means en Breast Cancer (k=2) - Proyección 3D PCA')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
plt.colorbar(scatter, ax=ax, label='Cluster (0/1)')
plt.show()