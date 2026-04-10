# Eleccion de Metodo Anticonceptivo — Pipeline Completo de Clasificacion

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![PyOD](https://img.shields.io/badge/PyOD-Outlier_Detection-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/Estado-Completo-green)

---

## Objetivo

Desarrollar un pipeline completo de **entendimiento, preparacion y clasificacion de datos** para predecir la eleccion del metodo anticonceptivo de mujeres casadas en Indonesia, aplicando tecnicas avanzadas de deteccion de outliers, balanceo de clases y reduccion de dimensionalidad.

---

## Dataset

**Contraceptive Method Choice Dataset** — UCI Machine Learning Repository
- Fuente: ucimlrepo — ID 30
- Origen: Encuesta Nacional de Prevalencia de Anticonceptivos de Indonesia (1987)
- Registros originales: 1.473 mujeres casadas no embarazadas
- Variables predictoras: 9 caracteristicas demograficas y socioeconomicas

| Clase | Descripcion |
|-------|-------------|
| 1 | No uso de metodo anticonceptivo |
| 2 | Uso de metodos a largo plazo |
| 3 | Uso de metodos a corto plazo |

### Variables del dataset

| Variable | Tipo | Descripcion |
|----------|------|-------------|
| wife_age | Numerica | Edad de la esposa |
| wife_edu | Categorica ordinal | Educacion esposa (1=baja, 4=alta) |
| husband_edu | Categorica ordinal | Educacion esposo (1=baja, 4=alta) |
| num_children | Numerica | Numero de hijos |
| wife_religion | Binaria | Religion (0=No islamica, 1=Islam) |
| wife_working | Binaria | Trabaja (0=Si, 1=No) |
| husband_occupation | Categorica | Ocupacion del esposo (1-4) |
| standard_of_living_index | Categorica ordinal | Nivel de vida (1=bajo, 4=alto) |
| media_exposure | Binaria | Exposicion a medios (0=Buena, 1=Mala) |

---

## Pipeline del Analisis

```
Carga de datos (ucimlrepo ID=30)
        │
        ▼
Etapa 1: Entendimiento de los datos
  - Distribucion de variables numericas (histograma + KDE)
  - Boxplots por clase objetivo (edad, hijos)
  - Graficos de frecuencia para variables categoricas
  - Matriz de correlacion (heatmap)
  - Verificacion de datos faltantes
  - Conteo por clase (balanceo)
        │
        ▼
Transformacion de variables categoricas
  - Binarizacion: wife_edu, husband_edu, standard_of_living_index
    (1 si valor > 1, 0 si valor = 1)
        │
        ▼
Deteccion de outliers multivariados
  - Isolation Forest (PyOD, contamination=5%)
  - Visualizacion de outliers en espacio PCA 2D
  - Eliminacion de 74 observaciones anomalas
        │
        ▼
Balanceo de clases
  - Undersampling a la clase minoritaria
  - Dataset balanceado con igual representacion por clase
        │
        ▼
Modelado: Regresion Logistica
  - Train/Test split 70/30 (estratificado)
  - Evaluacion: Accuracy, Precision, Recall
  - Matriz de confusion
  - Comparacion entre 3 escenarios
        │
        ▼
Reduccion de dimensionalidad (PCA)
  - Aplicado sobre dataset balanceado
  - n_components=0.90 (varianza explicada acumulada)
  - Evaluacion del modelo con componentes principales
```

---

## Tecnicas Aplicadas

### Deteccion de outliers
- **Isolation Forest (PyOD)** — metodo no supervisado basado en particiones aleatorias
- Contamination del 5% — detecta observaciones con combinaciones poco frecuentes
- Visualizacion de anomalias proyectadas en 2 componentes PCA

### Preparacion de datos
- **Binarizacion** de variables ordinales (educacion y nivel de vida)
- **Undersampling** para balanceo de clases
- **StandardScaler** para escalado previo a PCA y modelos

### Clasificacion
- **Regresion Logistica** multiclase (solver lbfgs)
- Metricas: Accuracy, Precision macro, Recall macro
- Reporte de clasificacion y matriz de confusion por escenario

### Reduccion de dimensionalidad
- **PCA** con umbral de varianza explicada del 90%

---

## Resultados Comparativos

| Escenario | Dataset | Accuracy |
|-----------|---------|----------|
| Original transformado | df_trans_cat | Base |
| Sin outliers | df_sin_outliers | Mejora moderada |
| **Balanceado** | **dataBal** | **Mejor resultado** |
| Balanceado + PCA | dataBal + PCA | Disminuye vs balanceado |

**Conclusion:** El balanceo de clases mejora significativamente las metricas macro de precision y recall. La aplicacion de PCA sobre el conjunto balanceado reduce el desempeno, lo que indica que las variables originales contienen informacion discriminativa relevante que se pierde con la transformacion. El modelo entrenado sobre el **dataset balanceado sin PCA** representa la mejor alternativa.

---

## Hallazgos exploratorios clave

- Las mujeres mas jovenes tienden a preferir metodos anticonceptivos a corto plazo
- A mayor nivel educativo de la esposa, menor numero promedio de hijos
- No se detectaron datos faltantes en el dataset
- 74 observaciones (5%) fueron identificadas como outliers multivariados por Isolation Forest
- No hay correlaciones superiores a 0.85 entre variables predictoras (sin multicolinealidad)

---

## Visualizaciones incluidas

- Histograma con curva KDE: distribucion de edad
- Boxplots: edad y numero de hijos vs metodo anticonceptivo
- Boxplot: numero de hijos vs metodo segun nivel educativo
- Graficos de frecuencia: todas las variables categoricas (grid 3x3)
- Heatmap: matriz de correlacion entre variables predictoras
- Scatter PCA 2D: visualizacion de outliers detectados (azul=normal, rojo=outlier)
- Matriz de confusion por escenario
- Tabla comparativa de metricas (Accuracy, Precision, Recall)

---

## Stack Tecnologico

```python
pandas           # Manipulacion y transformacion de datos
numpy            # Calculos numericos
scikit-learn     # LogisticRegression, PCA, StandardScaler, train_test_split
pyod             # Isolation Forest para deteccion de outliers
matplotlib       # Visualizaciones base
seaborn          # Visualizaciones estadisticas
copy             # Deep copy de DataFrames
ucimlrepo        # Carga directa del dataset UCI
```

---

## Como ejecutar el proyecto

```bash
# 1. Clonar el repositorio
git clone https://github.com/msorin81/clasificacion-metodos-anticonceptivos.git
cd clasificacion-metodos-anticonceptivos

# 2. Instalar dependencias
pip install pandas numpy scikit-learn pyod matplotlib seaborn ucimlrepo

# 3. Abrir el notebook
jupyter notebook "Caso_metodos_de_anticonceptivos_version_propia.ipynb"
```

> El dataset se descarga automaticamente mediante ucimlrepo. No es necesario descargar archivos adicionales.

---

## Estructura del repositorio

```
clasificacion-metodos-anticonceptivos/
│
├── Caso_metodos_de_anticonceptivos_version_propia.ipynb
└── README.md
```

---

## Autores

Proyecto desarrollado como ejercicio academico — Maestria en Ciencia de Datos

- Oscar Mauricio Montano
- Leimar Torres
- Juan Hurtado

**Oscar Mauricio Montano** — Analista de Datos | Power BI · SQL · Python · R

mbexcel@hotmail.com · Portafolio: https://github.com/msorin81
