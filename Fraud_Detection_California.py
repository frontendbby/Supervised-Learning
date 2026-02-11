import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Generacion de datos:
def generate_data(n=2000):
    """Simula un dataset"""
    np.random.seed(42)
    data = {
        'claim_id': [f'CLM_{i:04d}' for i in range(n)],
        'incident_state': np.random.choice(['CA', 'TX', 'NY', 'FL'], n, p=[0.4, 0.2, 0.2, 0.2]),
        'claim_amount': np.random.normal(5000, 3000, n).clip(500),
        'medical_fraud_score': np.random.beta(2, 5, n), # Puntuación de riesgo médico
        'days_since_policy': np.random.randint(0, 30, n),
        'vehicle_age': np.random.randint(0, 15, n)
    }
    df = pd.DataFrame(data)
    
    # Lógica de Negocio (SIU): Inyectamos el fraude basado en reglas
    def logic(row):
        p = 0.05 # Riesgo base
        if row['incident_state'] == 'CA': p += 0.10 # Foco en California
        if row['medical_fraud_score'] > 0.6: p += 0.20 # Anomalía médica
        if row['days_since_policy'] < 7: p += 0.15    # Fraude de "póliza fresca"
        return np.random.binomial(1, min(p, 1.0))

    df['is_fraud'] = df.apply(logic, axis=1)
    return df

# Ejecutamos la generación anterior y cargamos el dataset en memoria
df = generate_data()
print("Dataset generado y cargado en memoria.")


# Preparacion del modelo de Machine Learning
# El modelo no entiende nombres de ciudades, así que convertimos 'CA', 'TX', etc. a números con One-Hot Encoding (dummies)
df_ml = pd.get_dummies(df, columns=['incident_state'], drop_first=True)

# Definimos X (características) y Y (lo que queremos predecir)
X = df_ml.drop(['claim_id', 'is_fraud'], axis=1)
y = df_ml['is_fraud']

# Dividimos en Entrenamiento (80%) y Pruebas (20%) para evaluar honestamente y sin trampas!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenamiento del modelo que elegimos previamente: XGBoost, conocido por su rendimiento en clasificación de fraude
model = XGBClassifier(
    n_estimators=100, 
    max_depth=4,
    scale_pos_weight=9, # Balanceador de clases para detección de fraude!!! Sumamente importante en datasets desbalanceados
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)
print("Modelo XGBoost entrenado correctamente.")

# Agregamos visualizaciones:
plt.figure(figsize=(16, 6))

# Visualización A: Outliers y Mediana (metricas claves para detectar fraude)
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='is_fraud', y='medical_fraud_score', palette='Set2')
plt.title('Detección de Anomalías Médicas', fontsize=14)
plt.xticks([0, 1], ['Legítimo (0)', 'Fraude (1)'])
plt.annotate('Outliers: Casos extremos\nque la IA debe analizar', 
             xy=(0, 0.75), xytext=(0.2, 0.85),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Visualización B: Importancia de Variables para el modelo bajo la pregunta ¿Qué factores son más críticos para la IA al detectar fraude?
plt.subplot(1, 2, 2)
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', color='skyblue')
plt.title('¿Qué variables son más críticas para la IA?', fontsize=14)

plt.tight_layout()
plt.show()

y_pred = model.predict(X_test)
print("Reporte de inteligencia de fraude")
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
import pandas as pd

# Definimos los modelos con balance de peso para clase 1 que mas nos interesa (Fraude)
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost (Actual)": XGBClassifier(scale_pos_weight=9, eval_metric='logloss')
}

results = []

for name, model in models.items():
    # Entrenar
    model.fit(X_train, y_train)
    # Predecir
    preds = model.predict(X_test)
    # Medir
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    
    results.append({"Modelo": name, "Recall": rec, "Precision": prec})

# Mostrar comparativa
df_results = pd.DataFrame(results)
print("\n--- COMPARATIVA DE MODELOS ---")
print(df_results)

# Visualización de la batalla de modelos
df_results.set_index('Modelo').plot(kind='bar', figsize=(10, 6))
plt.title('Comparativa de Modelos: El desafío del Recall')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("Optimizacion de RF: busqueda de mejor recall")
# Paso A: Obtener probabilidades (en lugar de etiquetas fijas)
y_probs_rf = models["Random Forest"].predict_proba(X_test)[:, 1]

# Paso B: Bajamos el umbral a 0.25 (más sensibilidad)
nuevo_umbral = 0.25
y_pred_ajustado = (y_probs_rf >= nuevo_umbral).astype(int)

# Paso C: Comparativa de resultados
rec_original = df_results.loc[df_results['Modelo'] == 'Random Forest', 'Recall'].values[0]
rec_nuevo = recall_score(y_test, y_pred_ajustado)

print(f"Recall Original (Umbral 0.5): {rec_original:.4f}")
print(f"Nuevo Recall (Umbral {nuevo_umbral}): {rec_nuevo:.4f}")
print(f"Nueva Precision: {precision_score(y_test, y_pred_ajustado):.4f}")

# Visualización del impacto del Umbral
plt.figure(figsize=(8, 5))
plt.hist(y_probs_rf[y_test==0], bins=30, alpha=0.5, label='Legítimo', color='blue')
plt.hist(y_probs_rf[y_test==1], bins=30, alpha=0.5, label='Fraude', color='red')
plt.axvline(nuevo_umbral, color='black', linestyle='--', label=f'Nuevo Umbral ({nuevo_umbral})')
plt.title('Distribución de Probabilidades y el Nuevo Umbral')
plt.xlabel('Probabilidad de Fraude asignada por la IA')
plt.ylabel('Número de Casos')
plt.legend()
plt.show()

from imblearn.over_sampling import SMOTE
from collections import Counter

print("\n" + "="*40)
print("Aplicamos SMOTE para equilibrar las clases")
# Pasamos de 47 casos de fraude a tener los mismos que los legítimos (~353)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Distribución original: {Counter(y_train)}")
print(f"Distribución con SMOTE: {Counter(y_train_smote)}")

# 2. Re-entrenamos el Random Forest con los datos 'aumentados' nuestros datos ya estan balanceados para este punto
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)

# 3. Evaluación en el mundo real (Test Set original)
y_pred_smote = rf_smote.predict(X_test)

print("Reporte de clasificación después de SMOTE:")
print(classification_report(y_test, y_pred_smote))

# 4. Visualización de la 'Inyección' de Datos
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=y_train, palette='viridis')
plt.title('Antes de SMOTE (Desbalanceado)')

plt.subplot(1, 2, 2)
sns.countplot(x=y_train_smote, palette='viridis')
plt.title('Después de SMOTE (Equilibrado)')

plt.show()
