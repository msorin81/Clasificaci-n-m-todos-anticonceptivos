# 💊 Elección de Método Anticonceptivo — Clasificación con Regresión Logística

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/Estado-Completo-green)

---

## 🎯 Objetivo

Aplicar un pipeline completo de **preparación de datos y clasificación supervisada** para predecir la elección del método anticonceptivo de mujeres casadas en Indonesia, basándose en sus características demográficas y socioeconómicas.

---

## 📦 Dataset

**Contraceptive Method Choice Dataset** — UCI Machine Learning Repository
- Fuente: ucimlrepo — ID 30
- Origen: Encuesta Nacional de Prevalencia de Anticonceptivos de Indonesia (1987)
- Registros: 1.473 mujeres casadas, no embarazadas al momento de la entrevista
- Variables predictoras: 9 características demográficas y socioeconómicas
- Variable objetivo: método anticonceptivo elegido

| Clase | Descripción |
|-------|-------------|
| 1 | No uso de método anticonceptivo |
| 2 | Uso de métodos a largo plazo |
| 3 | Uso de métodos a corto plazo |

---

## 🔧 Pipeline del Análisis

```
Carga de datos (ucimlrepo ID=30)
        │
        ▼
Exploración y transformación de variables
  - Variables ordinales → category
  - Variables nominales → category
  - Variables binarias → category
        │
        ▼
Validación de calidad del dato
  - Sin datos faltantes
  - Sin valores atípicos relevantes
  - Correlación de Pearson (variables numéricas)
  - Correlación de Cramér (variables categóricas)
        │
        ▼
Codificación (One-Hot Encoding / dummies)
        │
        ▼
Balanceo de clases
  ┌────────────┐     ┌──────────────┐
  │   SMOTE    │     │ Undersampling│
  │(sobre-     │     │(sub-         │
  │ muestreo)  │     │ muestreo)    │
  └────────────┘     └──────────────┘
        │
        ▼
Reducción de dimensionalidad (PCA)
  - Scree Plot para selección de componentes
  - 10 componentes → 84-85% varianza explicada
        │
        ▼
Clasificación: Regresión Logística
  - Train/Test split 70/30 (estratificado)
  - Evaluación por Accuracy
  - Comparación entre 6 escenarios
```

---

## 🤖 Modelos y Técnicas Aplicadas

### Preparación de datos
- **One-Hot Encoding** para variables categóricas ordinales y nominales
- **SMOTE** (Synthetic Minority Oversampling Technique) para balanceo de clases
- **Undersampling** como técnica alternativa de balanceo
- **StandardScaler** para escalado previo a PCA y regresión

### Reducción de dimensionalidad
- **PCA** (Análisis de Componentes Principales) con selección mediante Scree Plot
- 10 componentes seleccionados capturando ~84-85% de la varianza total

### Clasificación
- **Regresión Logística** multiclase con scikit-learn
- Partición estratificada 70/30

---

## 📊 Resultados Comparativos

| Caso | Dataset | Accuracy |
|------|---------|----------|
| Caso 1 | Conjunto original transformado | 53.17% |
| Caso 5 | Balanceado con SMOTE (sin PCA) | **56.43%** ← Mejor resultado |
| Caso 6 | Balanceado con SMOTE + PCA | 47.09% |
| Anexo | Balanceado con Undersampling (sin PCA) | < SMOTE |
| Anexo | Balanceado con Undersampling + PCA | < SMOTE |

**Conclusión:** El balanceo con SMOTE sin PCA obtuvo el mejor desempeño. La aplicación de PCA redujo la accuracy al transformar las variables en componentes que pierden interpretabilidad para la regresión logística.

---

## 📈 Visualizaciones incluidas

- Boxplot: distribución de edad de la esposa y número de hijos
- Gráficos de barras: distribución de variables categóricas ordinales
- Gráficos de barras: distribución de variables binarias
- Scree Plot: varianza explicada acumulada por componentes PCA (SMOTE)
- Scree Plot: varianza explicada acumulada por componentes PCA (Undersampling)
- Tabla comparativa de accuracy por escenario

---

## 🛠️ Stack Tecnológico

```python
pandas           # Manipulación y transformación de datos
numpy            # Cálculos numéricos
scikit-learn     # LogisticRegression, PCA, StandardScaler, train_test_split
imbalanced-learn # SMOTE para balanceo de clases
scipy            # Chi2 para correlación de Cramér
matplotlib       # Visualizaciones base
seaborn          # Visualizaciones estadísticas
ucimlrepo        # Carga directa del dataset UCI
```

---

## 🚀 Cómo ejecutar el proyecto

```bash
# 1. Clonar el repositorio
git clone https://github.com/msorin81/clasificacion-metodos-anticonceptivos.git
cd clasificacion-metodos-anticonceptivos

# 2. Instalar dependencias
pip install pandas numpy scikit-learn imbalanced-learn scipy matplotlib seaborn ucimlrepo

# 3. Abrir el notebook
jupyter notebook "Caso_metodos_de_anticonceptivos.ipynb"
```

> **Nota:** El dataset se descarga automáticamente mediante `ucimlrepo`. No es necesario descargar archivos adicionales.

---

## 📁 Estructura del repositorio

```
clasificacion-metodos-anticonceptivos/
│
├── Caso_metodos_de_anticonceptivos.ipynb   # Notebook principal
└── README.md
```

---

## 👥 Autores

Proyecto desarrollado como ejercicio académico — Maestría en Ciencia de Datos

- Oscar Mauricio Montaño
- Leimar Torres
- Juan Hurtado

**Oscar Mauricio Montaño** — Analista de Datos | Power BI · SQL · Python · R

mbexcel@hotmail.com · Ver portafolio en GitHub: https://github.com/msorin81
