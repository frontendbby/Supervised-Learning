# 💻 Computer Pricing Analysis: A Multivariate Econometric Approach

Este repositorio contiene un análisis exhaustivo de los determinantes de precio en el mercado de hardware, utilizando modelos de **Regresión Lineal Múltiple (OLS)**. El objetivo es cuantificar el impacto de las especificaciones técnicas (RAM, HD, Velocidad) y factores de mercado (Tendencias y Publicidad) sobre el valor comercial de los equipos.

## 🔬 Rigor Metodológico y Arquitectura Estadística

El proyecto se estructura siguiendo los estándares de **Econometría Computacional**, dividiéndose en cuatro etapas críticas:

### 1. Análisis Exploratorio y Diagnóstico de Correlación

Se realizó un estudio de la distribución de precios y su colinealidad con variables técnicas.

* **Visualización Dinámica:** Uso de `pairplots` y matrices de correlación para identificar la fuerza de las relaciones lineales.
* **Interacción de Variables:** Evaluación del impacto de componentes premium y multimedia (CD-ROM) mediante gráficos de violín segmentados.

### 2. Modelado Predictivo OLS (Ordinary Least Squares)

Se desarrollaron e iteraron modelos de complejidad creciente:

* **Modelo Base:** Relación univariante Precio vs. Velocidad.
* **Modelo Multivariante de Hardware:** Integración de RAM, HD y tamaño de pantalla.
* **Modelo Extendido (Market Factors):** Inclusión de variables exógenas como la presión publicitaria (`ads`) y tendencias temporales (`trend`).

### 3. Validación de Supuestos de Gauss-Markov

Un análisis de regresión profesional requiere validar que los residuos sean ruido blanco. El proyecto incluye:

* **Homocedasticidad:** Análisis de dispersión de residuos frente a valores ajustados.
* **Normalidad:** Validación mediante **Gráficos Q-Q (Quantile-Quantile)**.
* **Multicolinealidad:** Cálculo del **VIF (Variance Inflation Factor)** para asegurar la independencia de las variables predictoras y evitar errores en los coeficientes.

### 4. Diagnóstico de Influencia y Outliers

Para garantizar la estabilidad del modelo, se implementaron técnicas de diagnóstico de puntos influyentes:

* **Distancia de Cook:** Identificación de observaciones dominantes que alteran desproporcionadamente la pendiente del modelo ().
* **Leverage (Hat Values):** Mapeo de residuos estudiantizados vs. apalancamiento para detectar puntos de alta influencia.

## 📈 Resultados Clave e Impacto

* **Poder Explicativo ():** El modelo extendido logra capturar una fracción significativa de la varianza del mercado, permitiendo predecir precios con un margen de error cuantificado.
* **Identificación de Observaciones Críticas:** Se detectaron las 5 unidades más influyentes que alteraban la generalización del modelo, permitiendo un ajuste más robusto.
* **Interpretación de Coeficientes:** El análisis permite entender cuánto valor monetario añade cada unidad adicional de RAM o GB de disco duro en el contexto del mercado analizado.

## 🛠️ Stack Tecnológico

* **Lógica Estadística:** `Statsmodels` (OLS, Influence diagnostics, VIF).
* **Análisis de Datos:** `Pandas`, `NumPy`.
* **Visualización:** `Seaborn`, `Matplotlib`.
* **Validación:** `SciPy.stats` (Q-Q Plots).

## 🚀 Instalación y Uso
1. Clonar el repositorio.
2. Instalar dependencias: `pip install pandas numpy matplotlib seaborn statsmodels scipy`.
3. Ejecutar el script: `python comp_pred.py`.
---

