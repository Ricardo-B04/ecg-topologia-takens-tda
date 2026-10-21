# Clasificación de ECG con análisis topológico

Experimento reproducible que compara un Random Forest entrenado con señales
ECG crudas y otro entrenado con imágenes de persistencia. Utiliza los latidos
MIT-BIH del [ECG Heartbeat Categorization Dataset de Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).
El análisis se limita a la clase **0 (N, normal)** y la **2 (V, latido
ectópico ventricular)**. La clase V no equivale por sí sola a un diagnóstico
de «arritmia severa».

## Requisitos e instalación

El experimento se ejecutó con Python **3.9.6**. En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` fija todas las versiones del entorno usado; para una
instalación menos estricta se puede usar `requirements-tda.txt`.

## Datos

Descarga `mitbih_train.csv` y `mitbih_test.csv` de la página de Kaggle y
colócalos en `data/`. Ambos archivos son CSV **sin encabezado**, con 187
valores de señal y una etiqueta por fila. También puedes usar la [CLI oficial
de Kaggle](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md),
una vez instalada y autenticada:

```bash
bash descargar_datos.sh
```

Verifica que son los mismos archivos usados para los resultados publicados:

```bash
python verificar_datos.py
```

Los hashes SHA-256 esperados están en `verificar_datos.py`. El dataset de
Kaggle indica licencia **desconocida**; por ello los CSV originales y los
artefactos que contienen señales no se distribuyen en este repositorio.

## Reproducir los cuatro pasos

Con el entorno activado y los datos verificados, ejecuta en este orden:

```bash
python 01_takens_ecg.py
python 02_vectorizacion_topologica.py --bottleneck-por-clase 10
python 03_benchmark_rf.py
python 04_invarianza_temporal.py
```

1. `01_takens_ecg.py` toma la fila 29 del archivo de entrenamiento, una señal
   de clase 0, y crea `01_takens_ecg.html`. Reconstruye la trayectoria 3D con
   dimensión 3 y retardo 3: \(p_i=(x_i,x_{i+3},x_{i+6})\), para
   \(i=0,\ldots,180\). Un latido aislado no garantiza un ciclo cerrado.
2. `02_vectorizacion_topologica.py` selecciona de entrenamiento 300 latidos
   por clase (semilla 42). Convierte cada señal en 181 puntos 3D mediante
   `SingleTakensEmbedding`, calcula persistencia Vietoris–Rips H0/H1 y crea
   imágenes de persistencia de 20 × 20 por dimensión. La opción indicada
   calcula además una matriz bottleneck exacta para 10 latidos por clase.
3. `03_benchmark_rf.py` entrena dos Random Forest de 300 árboles con los
   mismos 600 latidos: uno recibe 187 valores crudos y otro 800 valores de
   imagen topológica. Evalúa en 300 latidos por clase del archivo de prueba
   (semilla 43). La transformación topológica se ajusta solo con entrenamiento.
4. `04_invarianza_temporal.py` reutiliza esos modelos sin reentrenar. Mueve
   todas las señales de prueba hacia la derecha 0, 1, 3, 6, 12 y 24 muestras,
   añade ceros al inicio y recorta el final. Reconstruye la representación
   topológica en cada caso y vuelve a predecir.

Los archivos generados se guardan en `resultados/02_vectorizacion/`,
`resultados/03_benchmark/` y `resultados/04_invarianza_temporal/`. El último
directorio incluye `comparacion.csv`, predicciones por latido y una gráfica
interactiva autónoma `exactitud_desfase.html`.

## Resultados de referencia

Exactitud sobre la muestra balanceada de 600 latidos de prueba:

| Desfase (muestras) | ECG crudo | Imagen topológica |
| ---: | ---: | ---: |
| 0 | 92.3 % | 88.2 % |
| 1 | 80.5 % | 84.0 % |
| 3 | 74.8 % | 84.8 % |
| 6 | 77.5 % | 79.2 % |
| 12 | 67.3 % | 78.8 % |
| 24 | 64.0 % | 78.2 % |

La representación topológica mostró mayor **robustez relativa** ante estos
desfases, pero no invariancia exacta. Los ceros añadidos y el recorte cambian
parte de la nube de puntos y pueden modificar su diagrama. Los desfases están
en **muestras**: el dataset indica una frecuencia de 125 Hz, por lo que una
muestra corresponde a 8 ms, no a 1 ms. Las métricas reflejan una selección
balanceada; no representan la prevalencia de clases en el dataset. Los CSV
publicados no incluyen identificadores de paciente, por lo que este proyecto
no verifica una separación por paciente entre entrenamiento y prueba.

## Licencia

El código y la documentación de este repositorio se publican bajo [MIT](LICENSE).
Los datos de Kaggle y las dependencias tienen sus propias condiciones.
