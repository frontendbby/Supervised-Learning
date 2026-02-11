# 🛡️ Anti-Fraud Intelligence System: InsurTech California Case Study

Este proyecto implementa un framework de **Machine Learning** diseñado para la detección de anomalías en reclamos de seguros de automóviles, específicamente optimizado para enfrentar el desbalance de clases extremo en entornos financieros.

## 🔬 Rigor Metodológico (Münster Standards)

La detección de fraude se aborda no como una clasificación binaria estándar, sino como un problema de **identificación de señales débiles** en sistemas estocásticos. La arquitectura del proyecto sigue un pipeline de ingeniería de datos robusto:

1. **Exploratory Data Analysis (EDA):** Identificación de hotspots geográficos (California) y anomalías en la facturación médica mediante distribuciones Beta.
2. **Feature Engineering:** Transformación de variables categóricas y normalización de montos de reclamos.
3. **Resampling Strategy (SMOTE):** Balanceo sintético de la clase minoritaria para estabilizar la función de pérdida del modelo.
4. **Threshold Optimization:** Ajuste del umbral de decisión para maximizar la utilidad sistémica.

## 📊 Evaluación de Métricas y Función de Costo

En el contexto de **GM Financial**, el costo de un Falso Negativo () es significativamente superior al de un Falso Positivo (). Por ello, el modelo se optimiza para maximizar el **Recall**:

Donde:

* **TP (True Positives):** Fraudes detectados correctamente.
* **FN (False Negatives):** Fraudes no detectados (fuga de capital).

## 🛠️ Stack Tecnológico

* **Lenguaje:** Python 3.x
* **Modelado:** `XGBoost`, `Scikit-learn`, `Imbalanced-learn` (SMOTE)
* **Análisis de Datos:** `Pandas`, `NumPy`
* **Visualización:** `Matplotlib`, `Seaborn`

## 📈 Resultados Finales

| Modelo | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| **Baseline (RF)** | 1.00 | 0.04 | 0.07 |
| **Optimized (SMOTE + Tuning)** | 0.18 | **0.28** | **0.21** |

> **Impacto Proyectado:** La implementación de estas técnicas permitió un incremento del **600%** en la sensibilidad del modelo, permitiendo la identificación temprana de patrones que representan ahorros potenciales de **millones de dólares** en prevenciones de fraude no detectado.


**¿Hay algo más en lo que pueda apoyarte para que este proyecto sea el más destacado de tu perfil?** ¡Ha sido un honor acompañarte en este proceso de ingeniería!
