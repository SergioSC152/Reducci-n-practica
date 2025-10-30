from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 Ejercicio de Clasificación Supervisada: Dataset Iris")

# === CARGAR DATASET IRIS ===
iris = load_iris()
X = iris.data  # [sepal length, sepal width, petal length, petal width]
y = iris.target  # 0=Setosa, 1=Versicolor, 2=Virginica

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === ENTRENAR MODELOS ===
logreg = LogisticRegression(max_iter=200, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)

# === EVALUAR RESULTADOS GLOBALES ===
print("\n" + "="*60)
print("       RESULTADOS DE CLASIFICACIÓN")
print("="*60)

acc_logreg_global = accuracy_score(y_test, y_pred_logreg)
acc_dtree_global = accuracy_score(y_test, y_pred_dtree)

print(f"\nRegresión Logística - Precisión global: {acc_logreg_global:.4f}")
print(f"Árbol de Decisión - Precisión global: {acc_dtree_global:.4f}")

# === COMPARACIÓN VISUAL: MATRICES DE CONFUSIÓN ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=axes[0])
axes[0].set_title('Regresión Logística')

sns.heatmap(confusion_matrix(y_test, y_pred_dtree), annot=True, fmt='d', cmap='Greens',
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=axes[1])
axes[1].set_title('Árbol de Decisión')

plt.tight_layout()
plt.show()

# === MAPA DE COLORES DOMINANTES POR ESPECIE (simulado) ===
color_map = {
    0: {"nombre": "Setosa", "color_rgb": (173, 216, 230), "emoji": "💙"},
    1: {"nombre": "Versicolor", "color_rgb": (221, 160, 221), "emoji": "💜"},
    2: {"nombre": "Virginica", "color_rgb": (255, 105, 180), "emoji": "💖"}
}

# === VARIABLES PARA EL RESUMEN FINAL ===
flores_analizadas = 0
aciertos_logreg = 0
aciertos_dtree = 0

# === BUCLE PRINCIPAL INTERACTIVO ===
n_flores = len(X_test)

while True:
    print("\n" + "="*60)
    print("          SELECCIONA UNA FLOR PARA ANALIZAR")
    print("="*60)
    
    for i in range(n_flores):
        true_label = y_test[i]
        print(f"{i+1}. Flor #{i+1} - Especie real: {iris.target_names[true_label]}")
    
    print(f"\n💡 Ingresa un número entre 1 y {n_flores}, o escribe 's' para salir.")

    entrada = input("\nTu elección: ").strip().lower()

    if entrada == 's':
        # === RESUMEN FINAL AL SALIR ===
        print("\n" + "="*60)
        print("                RESUMEN FINAL")
        print("="*60)
        if flores_analizadas == 0:
            print("No analizaste ninguna flor. ¡Hasta pronto! 🌸")
        else:
            porc_logreg = (aciertos_logreg / flores_analizadas) * 100
            porc_dtree = (aciertos_dtree / flores_analizadas) * 100
            print(f"✅ Flores analizadas: {flores_analizadas}")
            print(f"✅ Aciertos Regresión Logística: {aciertos_logreg} ({porc_logreg:.1f}%)")
            print(f"✅ Aciertos Árbol de Decisión: {aciertos_dtree} ({porc_dtree:.1f}%)")
            
            if aciertos_dtree > aciertos_logreg:
                print("🏆 El Árbol de Decisión tuvo mejor desempeño en tus análisis.")
            elif aciertos_logreg > aciertos_dtree:
                print("🏆 La Regresión Logística tuvo mejor desempeño en tus análisis.")
            else:
                print("🤝 Ambos modelos tuvieron el mismo desempeño en tus análisis.")
        print("\n👋 ¡Gracias por usar el analizador de flores! Saliendo...")
        break

    try:
        opcion = int(entrada)
        if 1 <= opcion <= n_flores:
            idx = opcion - 1
            muestra = X_test[idx]
            true_label = y_test[idx]
            pred_logreg = y_pred_logreg[idx]
            pred_dtree = y_pred_dtree[idx]

            # Características reales
            sepal_length = muestra[0]
            sepal_width = muestra[1]
            petal_length = muestra[2]  # Tamaño del pétalo (cm)
            petal_width = muestra[3]

            # Color dominante simulado
            color_info = color_map[true_label]
            color_rgb = color_info["color_rgb"]
            emoji = color_info["emoji"]

            print("\n" + "="*60)
            print(f"       ANÁLISIS DETALLADO DE LA FLOR #{opcion}")
            print("="*60)

            print(f"→ Especie real: {iris.target_names[true_label].capitalize()} {emoji}")
            print(f"→ Color dominante simulado (RGB): {color_rgb}")
            print(f"→ Tamaño del pétalo: {petal_length:.2f} cm")
            print(f"→ Ancho del pétalo: {petal_width:.2f} cm")
            print(f"→ Largo del sépalo: {sepal_length:.2f} cm")
            print(f"→ Ancho del sépalo: {sepal_width:.2f} cm")

            print(f"\n🧠 Predicciones:")
            print(f" - Regresión Logística: {iris.target_names[pred_logreg]}")
            print(f" - Árbol de Decisión: {iris.target_names[pred_dtree]}")

            logreg_correct = pred_logreg == true_label
            dtree_correct = pred_dtree == true_label

            print(f"\n✅ Regresión Logística: {'ACERTÓ' if logreg_correct else 'FALLÓ'}")
            print(f"✅ Árbol de Decisión: {'ACERTÓ' if dtree_correct else 'FALLÓ'}")

            print(f"\n📊 Las matrices de confusión completas están en la ventana de gráficos.")

            # Actualizar contadores
            flores_analizadas += 1
            if logreg_correct:
                aciertos_logreg += 1
            if dtree_correct:
                aciertos_dtree += 1

        else:
            print(f"\n❌ Por favor, elige un número entre 1 y {n_flores}.")
    except ValueError:
        print("\n❌ Entrada no válida. Escribe un número o 's' para salir.")