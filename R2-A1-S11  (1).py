# Librerías
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============== CARGA Y PREPROCESAMIENTO ===============
# Cargar el conjunto de datos Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target  # etiquetas reales (0, 1, 2)

# Estandarizar las características (importante para K-Means y PCA)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=iris.feature_names)

# Reducción de dimensionalidad con PCA a 3 componentes (para visualización 3D)
pca = PCA(n_components=3)
X_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns=['PC1', 'PC2', 'PC3'])

# Mostrar varianza explicada por cada componente principal
print("Varianza explicada por cada componente PCA:")
print(pca.explained_variance_ratio_)
print(f"Varianza total explicada: {sum(pca.explained_variance_ratio_):.3f}\n")

# =============== EXPERIMENTACIÓN CON K = 2, 4, 5 ===============
# Valores de k a evaluar (según solicitud)
valores_k = [2, 4, 5]

# Diccionarios para almacenar los scores de cada k
sil_scores_original = {}  # para datos X_scaled
sil_scores_pca = {}       # para datos X_pca

print("Evaluación de Silhouette Score para diferentes valores de K:\n")
print("K | Silhouette (datos originales escalados) | Silhouette (datos PCA)")
print("-" * 70)

for k in valores_k:
    # 1. Aplicar K-Means a los datos originales estandarizados
    kmeans_orig = KMeans(n_clusters=k, random_state=42)
    clusters_orig = kmeans_orig.fit_predict(X_scaled)
    score_orig = silhouette_score(X_scaled, clusters_orig)
    sil_scores_original[k] = score_orig
    
    # 2. Aplicar K-Means a los datos transformados por PCA
    kmeans_pca = KMeans(n_clusters=k, random_state=42)
    clusters_pca = kmeans_pca.fit_predict(X_pca)
    score_pca = silhouette_score(X_pca, clusters_pca)
    sil_scores_pca[k] = score_pca
    
    # Mostrar resultados en tabla
    print(f"{k} | {score_orig:36.3f} | {score_pca:24.3f}")

# =============== CONCLUSIONES ===============
print("\n" + "="*60)
print("CONCLUSIONES")
print("="*60)

# Encontrar el mejor K según Silhouette Score en ambos casos
mejor_k_orig = max(sil_scores_original, key=sil_scores_original.get)
mejor_k_pca = max(sil_scores_pca, key=sil_scores_pca.get)

print(f"\n• Mejor K (datos originales escalados): K = {mejor_k_orig} (Score = {sil_scores_original[mejor_k_orig]:.3f})")
print(f"• Mejor K (datos PCA):                 K = {mejor_k_pca} (Score = {sil_scores_pca[mejor_k_pca]:.3f})")

# Interpretación esperada (basada en conocimiento del dataset Iris)
print("\n• Interpretación:")
print("  - El Silhouette Score mide la cohesión y separación de los clusters.")
print("  - Valores más cercanos a 1 indican clusters bien definidos.")
print("  - Aunque el dataset Iris tiene 3 clases reales, estamos evaluando solo K=2,4,5.")
print("  - Esperamos que K=2 tenga un score alto (porque dos especies son muy separables),")
print("    mientras que K=4 o K=5 podrían generar clusters artificiales y bajar el score.")