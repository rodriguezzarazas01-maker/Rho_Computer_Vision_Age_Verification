# Rho - Age Estimation With ResNet50

## Descripción del proyecto

La cadena de supermercados Good Seed desea explorar el uso de técnicas de visión artificial para ayudar a cumplir la normativa relacionada con la venta de bebidas alcohólicas. El objetivo principal consiste en desarrollar un modelo capaz de estimar la edad de una persona a partir de una fotografía facial, permitiendo identificar clientes potencialmente menores de edad y apoyar el proceso de verificación en los puntos de venta.

Para abordar esta problemática se implementó un modelo de Deep Learning basado en la arquitectura ResNet50 preentrenada con pesos de ImageNet, utilizando técnicas de transferencia de aprendizaje para mejorar el desempeño del sistema.

---

## Objetivo

Construir y evaluar un modelo de visión artificial capaz de predecir la edad de una persona a partir de una imagen facial.

---

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- TensorFlow
- Keras
- ResNet50
- Google Colab

---

## Dataset

El conjunto de datos contiene:

- 7,591 imágenes faciales.
- Edades comprendidas entre 1 y 100 años.
- Un archivo `labels.csv` con:
  - `file_name`
  - `real_age`

Estructura original:

```text
faces/
├── labels.csv
└── final_files/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

---

## Metodología

### 1. Análisis exploratorio de datos (EDA)

Se realizó un análisis exploratorio para comprender la estructura y calidad de los datos:

- Revisión de valores ausentes.
- Estadísticas descriptivas de la variable edad.
- Distribución de edades.
- Visualización de muestras de imágenes.
- Evaluación visual de iluminación, orientación y diversidad de rostros.

### Hallazgos principales

- El conjunto de datos contiene personas entre 1 y 100 años.
- Existe una mayor concentración de imágenes entre los 20 y 40 años.
- Se observa un ligero desbalance en edades extremas.
- Las imágenes presentan variabilidad en iluminación, orientación facial y accesorios, favoreciendo la capacidad de generalización del modelo.

---

## Preparación de datos

Para optimizar el consumo de memoria se utilizó:

```python
ImageDataGenerator()
```

con:

- Escalado de píxeles mediante `rescale=1./255`.
- Separación:
  - 80% entrenamiento.
  - 20% validación.

Las imágenes fueron redimensionadas a:

```text
224 x 224 píxeles
```

---

## Construcción del modelo

Se utilizó Transfer Learning mediante:

```python
ResNet50
```

Configuración utilizada:

- Pesos preentrenados de ImageNet.
- Eliminación de la capa superior original (`include_top=False`).
- GlobalAveragePooling2D.
- Capa densa final con una neurona para predicción de edad.
- Optimizador Adam.
- Función de pérdida MSE.
- Métrica MAE.

Arquitectura:

```text
Imagen
   ↓
ResNet50
   ↓
GlobalAveragePooling2D
   ↓
Dense(1)
   ↓
Edad estimada
```

---

## Resultados

El modelo fue entrenado durante 20 épocas utilizando GPU.

### Mejor resultado obtenido

| Métrica | Valor |
|----------|---------|
| Validation Loss | 112.6679 |
| Validation MAE | 8.1314 |

Este resultado fue alcanzado en la época 18.

### Resultado final

| Métrica | Valor |
|----------|---------|
| Training Loss | 15.9708 |
| Training MAE | 3.0936 |
| Validation Loss | 141.7143 |
| Validation MAE | 9.4365 |

---

## Interpretación de resultados

El modelo logró aprender patrones faciales asociados a la edad y reducir progresivamente el error durante el entrenamiento.

El mejor desempeño obtuvo un error absoluto medio de aproximadamente:

```text
8.13 años
```

Esto significa que las predicciones del modelo difieren de la edad real en cerca de ocho años en promedio.

Aunque el sistema muestra una capacidad razonable para estimar edades, el nivel de error observado indica que no debe sustituir completamente los métodos tradicionales de verificación de identidad.

---

## Conclusiones

- La arquitectura ResNet50 fue capaz de aprender características faciales relevantes para la estimación de edad.
- El modelo alcanzó un MAE de validación mínimo de 8.13 años.
- Los resultados sugieren que la visión artificial puede servir como herramienta de apoyo para la detección de clientes potencialmente menores de edad.
- El modelo no debería utilizarse como único mecanismo de validación para la venta de alcohol debido al margen de error observado.
- La solución puede aportar valor como sistema de alerta temprana para los empleados de Good Seed.

---

## Posibles mejoras futuras

- Aplicar Data Augmentation.
- Implementar Early Stopping.
- Ajustar hiperparámetros.
- Descongelar parcialmente capas de ResNet50 para Fine Tuning.
- Evaluar arquitecturas más recientes como EfficientNet o ConvNeXt.
- Incrementar la representación de grupos etarios menos frecuentes.

---

## Autor

**Santiago Rodríguez Zaraza**

Proyecto de Computer Vision desarrollado como parte de la formación en Ciencia de Datos.
