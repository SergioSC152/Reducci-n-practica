# Importar librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar datos
#https://www.kaggle.com/datasets/anishdevedward/loan-approval-dataset
df = pd.read_csv(r"C:\Users\zetom\Documents\INTRO ML\loan_approval.csv")

# Mostrar información inicial
print("Primeras filas del dataset:")
print(df.head(), "\n")

print("Resumen estadístico:")
print(df.describe(), "\n")

print("Valores nulos:")
print(df.isnull().sum(), "\n")

# Seleccionar variables relevantes
X = df[['income', 'credit_score', 'loan_amount', 'years_employed', 'points']]
y = df['loan_approved'].astype(int)  # Convertir True/False a 1/0

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Crear y entrenar el modelo
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar resultados
print("Precisión del modelo:", round(accuracy_score(y_test, y_pred), 3))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# Ejemplo de predicción
nuevo_cliente = pd.DataFrame({
    'income': [55000],
    'credit_score': [700],
    'loan_amount': [20000],
    'years_employed': [10],
    'points': [55]
})

prediccion = modelo.predict(nuevo_cliente)
probabilidad = modelo.predict_proba(nuevo_cliente)

print("\nPredicción para nuevo cliente:", "Aprobado ✅" if prediccion[0] == 1 else "No aprobado ❌")
print("Probabilidad:", probabilidad[0])
