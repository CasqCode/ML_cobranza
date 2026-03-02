# Documentación – Notebook de Exploración de Datos (EDA)

## 1. Objetivo del notebook
Este notebook tiene como objetivo realizar una **Exploración de Datos (EDA)** inicial sobre el dataset, con el fin de:
- Validar calidad de la información
- Identificar problemas de datos (nulos, duplicados, tipos incorrectos)
- Comprender la distribución de variables numéricas y categóricas
- Analizar la relación de las variables con la variable objetivo **`pago`**
- Obtener insights preliminares para la fase de modelado

---

## 2. Carga de librerías y configuración de rutas
Se importan las librerías fundamentales para análisis de datos:
- `pandas`, `numpy` para manipulación de datos
- `os`, `pathlib` para manejo de rutas
- `matplotlib` / `seaborn` para visualización

Se define dinámicamente el **directorio del proyecto**, permitiendo que el notebook sea portable entre entornos.

---

## 3. Normalización de nombres de columnas
```python
.str.lower()
.str.strip()
.str.replace(' ', '_')
```

**Objetivo:**
- Estandarizar los nombres de columnas
- Evitar errores posteriores en consultas, modelado o joins

**Resultado:**
Todas las columnas quedan en minúscula, sin espacios y con formato consistente.

---

## 4. Revisión de tipos de datos
```python
print(df.dtypes)
```

**Objetivo:**
- Verificar que cada variable tenga el tipo de dato adecuado
- Detectar variables numéricas cargadas como texto

Esta validación es crítica antes de cualquier análisis estadístico o modelado.

---

## 5. Análisis de valores nulos
```python
nulos = df.isnull().sum()
```

**Objetivo:**
- Identificar columnas con información faltante
- Priorizar acciones de limpieza o imputación

El resultado permite decidir:
- Eliminación de variables
- Creación de flags de información faltante
- Estrategias de imputación

---

## 6. Detección de duplicados
Se calcula el número de duplicados por columna.

**Objetivo:**
- Detectar problemas de calidad
- Validar llaves naturales o identificadores

---

## 7. Estadísticas descriptivas – Variables numéricas
```python
df.describe().T
```

**Objetivo:**
- Analizar distribución, rangos y outliers
- Identificar escalas y posibles transformaciones

Incluye métricas como:
- Media
- Mediana
- Percentiles
- Valores mínimos y máximos

---

## 8. Estadísticas descriptivas – Variables categóricas
```python
df.describe(include='object').T
```

**Objetivo:**
- Identificar cardinalidad
- Revisar valores dominantes
- Detectar posibles errores de codificación

---

## 9. Cardinalidad de variables categóricas
Se imprime el conteo de los valores más frecuentes por variable categórica.

**Objetivo:**
- Identificar variables de alta cardinalidad
- Evaluar necesidad de binning o encoding especial

---

## 10. Distribución de la variable objetivo (`pago`) por mes
```python
df.groupby('mes')['pago'].mean()
```

**Objetivo:**
- Analizar comportamiento temporal del pago
- Detectar estacionalidades o tendencias

La visualización permite validar estabilidad del fenómeno a lo largo del tiempo.

---

## 11. Selección de variables numéricas
```python
df.select_dtypes(include='number')
```

**Objetivo:**
- Definir subconjunto de variables aptas para análisis de correlación

---

## 12. Matriz de correlación
```python
df_num.corr()
```

**Objetivo:**
- Analizar relaciones lineales entre variables numéricas
- Detectar multicolinealidad

La matriz se visualiza mediante un heatmap para facilitar interpretación.

---

## 13. Correlación con la variable objetivo (`pago`)
```python
df_num.corr()['pago']
```

**Objetivo:**
- Identificar variables con mayor relación directa con el pago
- Priorizar features para el modelado

Se ordenan las correlaciones de mayor a menor impacto.

---

## 14. Visualización de correlación con `pago`
Se genera un gráfico de barras excluyendo la variable objetivo.

**Objetivo:**
- Facilitar interpretación ejecutiva
- Comparar fuerza relativa de cada variable

---

## 15. Análisis de variables categóricas vs pago
Para cada variable categórica:
```python
df.groupby(col)['pago'].mean()
```

**Objetivo:**
- Calcular la tasa de pago por categoría
- Identificar segmentos de alto y bajo desempeño

Este análisis es clave para:
- Feature engineering
- Segmentación de clientes
- Definición de reglas de negocio

---

## 16. Visualización de tasa de pago por categoría
Se generan gráficos para cada variable categórica.

**Objetivo:**
- Detectar patrones visuales
- Comunicar resultados a negocio

---

## 17. Conclusión
Este notebook cumple la función de **base analítica** para:
- Limpieza de datos
- Ingeniería de variables
- Selección de features
- Construcción de modelos predictivos

Sirve como insumo directo para fases posteriores de **modelado, validación y producción**.

---
