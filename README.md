# Pipeline End-to-End — Modelo de Propensión de Pago

**Autor:** Emerson Aguilar Cruz  
**Lenguaje:** Python 3  
**Dominio:** Data Science · Riesgo Crediticio · Cobranza  
**Tipo:** Pipeline productivo con clase reutilizable, evaluación y exportación  

---

## Descripción

Este proyecto implementa un pipeline completo de Machine Learning para **predecir la probabilidad de pago de clientes en cartera de cobranza**. El objetivo es transformar la gestión reactiva en una gestión preventiva y priorizada, usando modelos que aprenden del comportamiento histórico del cliente.

El diseño respeta la causalidad temporal (sin data leakage), maneja desbalance extremo de clases y genera features explicables alineadas al negocio.

---

## Resultados del Modelo

| Métrica | Regresión Logística (Baseline) | XGBoost (Final) |
|---|---|---|
| AUC-ROC Validación | 0.7648 | **0.7979** |
| AUC-ROC Test | — | 0.6458 |
| F1-Score Test | — | 0.0891 |
| Recall Test | — | 0.4082 |
| Accuracy Test | — | 0.8977 |

**Distribución del dataset tras el split temporal:**

| Conjunto | Registros | Tasa de pago |
|---|---|---|
| Train | 17,575 | 1.67% |
| Validación | 4,040 | 3.27% |
| Test | 7,998 | 1.23% |

**Desbalance de clases antes/después de SMOTE:**

| Clase | Antes | Después |
|---|---|---|
| 0 (no paga) | 17,282 | 17,282 |
| 1 (paga) | 293 | 5,184 |

**Top 5 features más importantes (XGBoost):**

| Feature | Importancia |
|---|---|
| duracion_llamadas_ultimos_6meses | 0.2328 |
| sin_pago_previo | 0.1466 |
| contacto_total | 0.0813 |
| sin_historial_pago | 0.0603 |
| sin_antiguedad_conocida | 0.0541 |

> El modelo fue evaluado en 84 variables. En el top 10% de clientes con mayor score de propensión, la tasa de pago real alcanzó el **5.00%** sobre una base de 800 clientes — más de 4× la tasa promedio del conjunto de test (1.23%).

---

## Arquitectura del Pipeline

```
Carga de datos (Excel)
        ↓
Limpieza y estandarización
        ↓
Features de información faltante
        ↓
Features de negocio (ratios, flags, interacciones)
        ↓
Binning / Intervalos
        ↓
Features temporales (lags, rolling, deltas, contexto)
        ↓
Split temporal estricto (Train / Val / Test)
        ↓
Encoding categórico (OHE + Target Encoding)
        ↓
Balanceo de clases (SMOTE)
        ↓
Entrenamiento (LR Baseline → XGBoost)
        ↓
Optimización de threshold
        ↓
Evaluación final + estabilidad temporal
        ↓
Exportación CSV con scores de propensión
```

---

## Estructura del Proyecto

```
├── src/
│   └── _cls_ml_prueba_ds.py   # Clase principal PruebasDs
├── data/
│   └── predicciones_test.csv     # Output del modelo (no incluye datos de entrada)
├── notebooks/
│   └── _exploracion_datos.ipynb  # Análisis exploratorio
├── requirements.txt
└── README.md
```

> ⚠️ El archivo de datos de entrada (`PruebaDS.xlsx`) no está incluido en el repositorio por contener información sensible.

---

## Instalación

```bash
git clone https://github.com/tu-usuario/tu-repo.git](https://github.com/CasqCode/ML_cobranza.git
cd tu-repo
pip install -r requirements.txt
```

---

## Dependencias

```
pandas
numpy
scikit-learn
imbalanced-learn
category_encoders
xgboost
openpyxl
```

---

## Ejecución

```bash
python src/_cls_ml_prueba_ds.py
```

El pipeline corre de forma secuencial desde el bloque `if __name__ == "__main__":` y genera automáticamente el archivo `predicciones_test.csv` con la probabilidad estimada de pago por cliente.

---

## Clase Principal: `MlPruebaDS`

Toda la lógica está encapsulada en una clase reutilizable que permite controlar cada etapa de forma independiente o ejecutar el pipeline completo.

| Método | Descripción |
|---|---|
| `load_datos()` | Carga el archivo Excel de entrada |
| `data_cleaning()` | Limpieza, fechas, normalización de género |
| `crear_features_nulos()` | Flags binarios para información faltante |
| `crear_features_negocio()` | Ratios, interacciones, señales de riesgo |
| `crear_intervalos()` | Binning de variables continuas |
| `crear_features_temporales()` | Lags, rolling, deltas, contexto |
| `split_temporal()` | División respetando causalidad temporal |
| `aplicar_encoding()` | OHE para baja cardinalidad, Target Encoding para alta |
| `preparar_X_y()` | Separación de features y target |
| `aplicar_balanceo()` | SMOTE con estrategia configurable |
| `entrenar_baseline()` | Regresión Logística con StandardScaler |
| `entrenar_xgboost()` | XGBoost con early stopping y AUC como métrica |
| `optimizar_threshold()` | Búsqueda del umbral óptimo (F1, Precision o Recall) |
| `evaluar_modelo_completo()` | Métricas completas + estabilidad temporal por mes |
| `mostrar_feature_importance()` | Top N features por importancia |
| `analizar_clientes_alta_probabilidad()` | Segmento de mayor propensión |
| `exportar_predicciones()` | CSV con ID, mes, pago real, score y predicción |

---

## Decisiones de Diseño

**Split temporal en lugar de split aleatorio**  
En datos con secuencia temporal, un split aleatorio genera data leakage: el modelo aprende de eventos futuros. El split temporal garantiza que las métricas reflejen el rendimiento real esperado en producción.

**SMOTE solo sobre el conjunto de entrenamiento**  
El balanceo se aplica exclusivamente al train, nunca a validación ni test. Aplicarlo a todos los conjuntos inflaría artificialmente las métricas.

**Target Encoding solo con fit en train**  
El encoder se ajusta únicamente con los datos de entrenamiento y se aplica (transform) sobre validación y test, evitando filtración de información del target.

**scale_pos_weight en XGBoost**  
Además de SMOTE, se configura `scale_pos_weight` como la razón entre clases negativa/positiva, reforzando la sensibilidad del modelo hacia la clase minoritaria.

---

## Contexto del Problema

La tasa de pago real en el conjunto de test es del **1.23%** — un desbalance de aproximadamente 80:1 entre clientes que no pagan y los que sí pagan. Este escenario es común en carteras de cobranza y requiere un tratamiento explícito para que el modelo no colapse prediciendo siempre la clase mayoritaria.

El objetivo no es maximizar accuracy (trivialmente alto al predecir siempre "no paga"), sino identificar correctamente los clientes con mayor probabilidad de pago para priorizar la gestión.

---

## Licencia


Este proyecto es de uso personal y educativo. El código puede ser revisado libremente. No se permite su reproducción, distribución ni uso comercial sin autorización del autor.
