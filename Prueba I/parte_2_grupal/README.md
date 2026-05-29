# Proyecto INF8090 — Computación Paralela y Distribuida

Repositorio de trabajo para una prueba/proyecto de **computación paralela y distribuida**, con implementaciones en C++, OpenMP y Python. El contenido principal está organizado en dos líneas:

1. Una presentación HTML sobre **cracking masivo de hashes** y comparación de arquitecturas paralelas.
2. La carpeta `parte_2_grupal/`, que contiene ejercicios de paralelización, benchmarking e informe en LaTeX/PDF.

---

## Estructura general

```text
Prueba/
├── Proyecto_integrador.html
└── parte_2_grupal/
    ├── Ejercicio_1.cpp
    ├── Ejercicio_2/
    │   ├── benchmark.py
    │   └── generate_logs.py
    ├── Ejercicio_3.py
    ├── Informe-grupal.tex
    ├── Informe-grupal.pdf
    └── OpenMP/
        ├── Leeme.txt
        ├── make_env.exe
        ├── openmp_ejemplos/
        │   ├── Ejercicio_3.cpp
        │   ├── embeddings_topk_openmp.cpp
        │   ├── embeddings_topk_openmp_simple.cpp
        │   ├── primos_muy_pesado.c
        │   ├── primos_pesado.c
        │   ├── suma_50M.c
        │   └── archivos generados de compilación
        └── w64devkit/
            └── toolchain portable para Windows
```

---

## Contenido principal

### `Proyecto_integrador.html`

Presentación web en formato de diapositivas sobre **Cracking Masivo de Hashes**. Explica el problema como una tarea `CPU-bound` y altamente paralelizable, usando como referencia el dataset `rockyou.txt`. La presentación compara enfoques secuenciales, OpenMP, CUDA y OpenCL.

Características:

- HTML estático.
- Estilo con Tailwind CSS mediante CDN.
- Gráficos con Chart.js mediante CDN.
- Diseño tipo presentación con scroll vertical.

Para abrirlo:

```bash
# Linux/macOS
xdg-open Proyecto_integrador.html

# Windows
start Proyecto_integrador.html
```

Nota: como usa CDN para Tailwind y Chart.js, necesita conexión a internet para verse correctamente si esos recursos no están cacheados.

---

## Parte 2 grupal

### Ejercicio 1 — Normalización Z-Score con OpenMP

Archivo principal:

```text
parte_2_grupal/Ejercicio_1.cpp
```

Este programa implementa una normalización **Z-Score** sobre una matriz binaria de `double` con 16 columnas. El procesamiento se divide en tres fases paralelizadas con OpenMP:

1. Cálculo de media por columna ignorando valores `NaN`.
2. Cálculo de desviación estándar por columna ignorando valores `NaN`.
3. Normalización Z-Score y conteo de valores atípicos con `|z| > 3`.

El archivo esperado por defecto es:

```text
dataset_matriz_X.bin
```

Formato esperado:

- Archivo binario plano.
- Tipo de dato: `double`.
- Matriz de dimensión `nrows × 16`.
- Valor por defecto de filas: `50,000,000`.

Compilación:

```bash
g++ -O3 -fopenmp -march=native Ejercicio_1.cpp -o ejercicio1
```

Ejecución con valores por defecto:

```bash
./ejercicio1
```

Ejecución indicando archivo y número de filas:

```bash
./ejercicio1 dataset_matriz_X.bin 50000000
```

En Windows usando `w64devkit`:

```bash
g++ -O3 -fopenmp Ejercicio_1.cpp -o ejercicio1.exe
./ejercicio1.exe dataset_matriz_X.bin 50000000
```

Limitación importante: el ZIP no incluye `dataset_matriz_X.bin`, por lo tanto este ejercicio compila, pero no puede ejecutarse con el dataset completo sin generar o copiar ese archivo binario.

---

### Ejercicio 2 — Scripts de benchmark

Carpeta:

```text
parte_2_grupal/Ejercicio_2/
```

Archivos encontrados:

```text
benchmark.py

generate_logs.py
```

Estado actual:

- Ambos archivos están vacíos.
- No existe lógica implementada para benchmarking o generación de logs.

Si el ejercicio 2 es parte de la entrega evaluada, esta sección está incompleta y debe implementarse antes de entregar el proyecto.

---

### Ejercicio 3 — Cálculo de similitud por bloques para embeddings

Este ejercicio aparece en dos variantes: una implementación en Python y varias implementaciones en C++ con OpenMP.

---

## Ejercicio 3 en Python

Archivo:

```text
parte_2_grupal/Ejercicio_3.py
```

El programa genera embeddings sintéticos normalizados y calcula similitud coseno por bloques para obtener el **Top-K de vectores más similares** por cada vector. Usa:

- `NumPy` para operaciones matriciales.
- `multiprocessing` para paralelizar por procesos.
- `shared_memory` para compartir la matriz de embeddings entre workers.
- `ProcessPoolExecutor` para repartir rangos de filas.

Dependencias:

```bash
pip install numpy
```

Ejecución por defecto:

```bash
python Ejercicio_3.py
```

Parámetros disponibles:

```bash
python Ejercicio_3.py --n 20000 --d 128 --block 1024 --k 10 --workers 1 2 4 8
```

Significado de los parámetros:

| Parámetro | Descripción | Valor por defecto |
|---|---:|---:|
| `--n` | Número de vectores | `20000` |
| `--d` | Dimensión de cada embedding | `128` |
| `--block` | Tamaño del bloque de columnas | `1024` |
| `--k` | Número de vecinos más similares | `10` |
| `--workers` | Lista de procesos a evaluar | `1 2 4 8` |

Ejemplo reducido para prueba rápida:

```bash
python Ejercicio_3.py --n 1000 --d 64 --block 256 --k 10 --workers 1 2 4
```

Salida esperada:

- Estimación de memoria.
- Tiempo de ejecución por cantidad de workers.
- Speedup.
- Eficiencia.
- Throughput en comparaciones por segundo.
- Top-K del vector 0.

Advertencia técnica: el algoritmo exacto es de complejidad `O(N²D)`. Para `N=20000`, se evalúan aproximadamente `399,980,000` comparaciones, por lo que puede tardar bastante dependiendo del equipo. Para validar funcionamiento, conviene partir con `N=1000` o menos.

Otra observación: la estimación de memoria del bloque en el script usa `B × B`, pero cada worker calcula `Xi @ Xj.T`, donde `Xi` corresponde al rango completo de filas asignado al worker. En la práctica, el bloque temporal puede acercarse a `(N / workers) × B`, no necesariamente a `B × B`.

---

## Ejercicio 3 en C++ con OpenMP

Carpeta:

```text
parte_2_grupal/OpenMP/openmp_ejemplos/
```

Archivos relevantes:

```text
Ejercicio_3.cpp
embeddings_topk_openmp.cpp
embeddings_topk_openmp_simple.cpp
```

### `embeddings_topk_openmp.cpp`

Implementación compacta de Top-K de similitud coseno con OpenMP. Genera embeddings sintéticos, los normaliza y calcula los vecinos más similares sin almacenar la matriz completa `N × N`.

Compilación:

```bash
g++ -O3 -march=native -fopenmp -std=c++17 embeddings_topk_openmp.cpp -o embeddings_topk_openmp
```

Ejecución por defecto:

```bash
./embeddings_topk_openmp
```

Ejecución con parámetros:

```bash
./embeddings_topk_openmp 20000 128 10 512 1024 1
```

Formato de parámetros:

```text
./embeddings_topk_openmp <N> <D> <K> <ROW_BLOCK> <COL_BLOCK> <BENCHMARK_MODE>
```

Ejemplo reducido:

```bash
./embeddings_topk_openmp 1000 64 10 256 256 1
```

Salida principal:

- Configuración usada.
- Estimación de memoria.
- Validación de norma 1.
- Tabla benchmark para 1, 2, 4 y 8 hilos.
- Top-K del vector 0.

### `embeddings_topk_openmp_simple.cpp`

Versión alternativa y más simple del mismo enfoque. Usa una estructura más directa para entender el cálculo por bloques y el benchmark.

Compilación:

```bash
g++ -O3 -march=native -fopenmp -std=c++17 embeddings_topk_openmp_simple.cpp -o embeddings_topk_openmp_simple
```

Ejecución:

```bash
./embeddings_topk_openmp_simple 20000 128 10 512 1024 1
```

### `Ejercicio_3.cpp`

Versión más extensa y explicativa. Incluye diagnóstico, comparación de estrategias y justificación de la estrategia seleccionada.

Estrategias comparadas:

1. Baseline secuencial.
2. Estrategia naive paralela.
3. Estrategia por bloques con triangular y locks.
4. Estrategia por bloques con Top-K privado por hilo.

Compilación:

```bash
g++ -O3 -march=native -fopenmp -std=c++17 Ejercicio_3.cpp -o ejercicio3
```

Ejecución:

```bash
./ejercicio3
```

Ejecución indicando cantidad de hilos y tamaño de bloque:

```bash
./ejercicio3 8 512
```

Formato:

```text
./ejercicio3 <num_threads> <block_size>
```

---

## Informe grupal

Archivos:

```text
parte_2_grupal/Informe-grupal.tex
parte_2_grupal/Informe-grupal.pdf
```

El informe en LaTeX documenta los ejercicios grupales, incluyendo diagnóstico, estrategia de paralelización, métricas de rendimiento, tablas y discusión técnica.

Compilación del informe:

```bash
pdflatex Informe-grupal.tex
pdflatex Informe-grupal.tex
```

Si se usa VS Code, se recomienda instalar:

- LaTeX Workshop.
- MiKTeX o TeX Live.

---

## Requisitos

### Para C++ / OpenMP

Linux:

```bash
sudo apt update
sudo apt install g++ make
```

Compilación general:

```bash
g++ -O3 -fopenmp -std=c++17 archivo.cpp -o programa
```

Windows:

El ZIP incluye una carpeta `w64devkit/` con herramientas portables para compilar C/C++ en Windows. También se incluye `OpenMP/Leeme.txt`, que explica cómo configurar VS Code con ese entorno.

Comando típico en Windows con `w64devkit`:

```bash
g++ -O3 -fopenmp -std=c++17 archivo.cpp -o programa.exe
```

### Para Python

Versión recomendada:

```text
Python 3.10+
```

Dependencias:

```bash
pip install numpy
```

---

## Pruebas rápidas recomendadas

Para evitar ejecuciones demasiado largas, probar primero con tamaños reducidos.

Python:

```bash
cd parte_2_grupal
python Ejercicio_3.py --n 500 --d 32 --block 128 --k 5 --workers 1 2
```

C++ OpenMP:

```bash
cd parte_2_grupal/OpenMP/openmp_ejemplos
g++ -O3 -fopenmp -std=c++17 embeddings_topk_openmp.cpp -o embeddings_topk_openmp
./embeddings_topk_openmp 500 32 5 128 128 1
```

Ejercicio 1 solo puede ejecutarse si existe el archivo binario de entrada:

```bash
cd parte_2_grupal
g++ -O3 -fopenmp Ejercicio_1.cpp -o ejercicio1
./ejercicio1 dataset_matriz_X.bin 50000000
```

---

## Archivos generados y limpieza recomendada

El ZIP contiene varios archivos compilados o generados que no deberían versionarse en Git:

```text
*.exe
*.obj
*.pdb
*.ilk
build/
w64devkit/
.vscode/
```

También contiene un toolchain completo (`w64devkit`), lo que explica el tamaño elevado del ZIP. Para una entrega limpia o repositorio GitHub, se recomienda conservar solo:

```text
Proyecto_integrador.html
parte_2_grupal/Ejercicio_1.cpp
parte_2_grupal/Ejercicio_3.py
parte_2_grupal/Informe-grupal.tex
parte_2_grupal/Informe-grupal.pdf
parte_2_grupal/OpenMP/Leeme.txt
parte_2_grupal/OpenMP/openmp_ejemplos/*.cpp
parte_2_grupal/OpenMP/openmp_ejemplos/*.c
README.md
```

`.gitignore` sugerido:

```gitignore
# Binarios y compilación
*.exe
*.out
*.obj
*.o
*.pdb
*.ilk
build/

# Toolchain portable pesado
w64devkit/

# VS Code local
.vscode/

# Datasets pesados
*.bin
*.csv
*.npy
*.npz

# Python
__pycache__/
*.pyc
.venv/
venv/

# LaTeX auxiliares
*.aux
*.log
*.toc
*.out
*.synctex.gz
```

---

## Estado de revisión del paquete

Revisión realizada sobre el ZIP:

- `Ejercicio_1.cpp` compila correctamente con `g++` y OpenMP.
- `Ejercicio_3.py` ejecuta correctamente con parámetros pequeños.
- `embeddings_topk_openmp.cpp` compila correctamente con `g++`, OpenMP y C++17.
- `embeddings_topk_openmp_simple.cpp` compila correctamente con `g++`, OpenMP y C++17.
- `Ejercicio_3.cpp` compila correctamente con `g++`, OpenMP y C++17.
- `benchmark.py` y `generate_logs.py` están vacíos.
- El dataset binario requerido por `Ejercicio_1.cpp` no está incluido.
- El ZIP incluye muchos binarios y herramientas externas, por lo que no está optimizado para entrega limpia ni para repositorio.

---

## Observaciones técnicas finales

El proyecto está orientado a evaluar rendimiento paralelo en problemas de alto costo computacional. La parte más sólida del ZIP está en el **Ejercicio 3**, porque contiene versiones en Python y C++ para calcular similitud entre embeddings sin construir explícitamente toda la matriz `N × N`.

El punto más débil es el **Ejercicio 2**, porque los scripts están vacíos. También falta el dataset binario para ejecutar el **Ejercicio 1** en su tamaño completo. Si esto se entrega como proyecto final, conviene limpiar binarios, documentar cómo generar datos de prueba y completar o eliminar la carpeta del Ejercicio 2.
