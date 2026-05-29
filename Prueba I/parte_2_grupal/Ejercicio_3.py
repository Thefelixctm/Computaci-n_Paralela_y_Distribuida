import os

# Evita que NumPy use varios hilos internamente.
# Así el speedup medido viene de multiprocessing y no de BLAS oculto.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import time
import math
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def generar_embeddings(N, D, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(N, D)).astype(np.float32)

    normas = np.linalg.norm(X, axis=1, keepdims=True)
    normas[normas == 0] = 1.0

    X = X / normas
    return X.astype(np.float32)


def estimar_memoria(N, D, B, K):
    bytes_X = N * D * 4
    bytes_matriz_float32 = N * N * 4
    bytes_matriz_float64 = N * N * 8
    bytes_bloque = B * B * 4
    bytes_top_values = N * K * 4
    bytes_top_indices = N * K * 4

    memoria_bloques = bytes_X + bytes_bloque + bytes_top_values + bytes_top_indices

    print("\n=== Estimación de memoria ===")
    print(f"Matriz X float32: {bytes_X / 1024**2:.2f} MB")
    print(f"Matriz completa similitud float32: {bytes_matriz_float32 / 1024**3:.2f} GB")
    print(f"Matriz completa similitud float64: {bytes_matriz_float64 / 1024**3:.2f} GB")
    print(f"Bloque temporal aprox. BxB: {bytes_bloque / 1024**2:.2f} MB")
    print(f"Top values: {bytes_top_values / 1024**2:.2f} MB")
    print(f"Top indices: {bytes_top_indices / 1024**2:.2f} MB")
    print(f"Memoria estrategia por bloques aprox.: {memoria_bloques / 1024**2:.2f} MB")


def dividir_rangos(N, workers):
    rangos = []
    base = N // workers
    resto = N % workers

    inicio = 0
    for w in range(workers):
        tam = base + (1 if w < resto else 0)
        fin = inicio + tam
        rangos.append((inicio, fin))
        inicio = fin

    return rangos


def worker_topk(args):
    row_start, row_end, N, D, B, K, shm_name = args

    shm = shared_memory.SharedMemory(name=shm_name)
    X = np.ndarray((N, D), dtype=np.float32, buffer=shm.buf)

    row_count = row_end - row_start
    Xi = X[row_start:row_end]

    top_values = np.full((row_count, K), -np.inf, dtype=np.float32)
    top_indices = np.full((row_count, K), -1, dtype=np.int32)

    row_ids = np.arange(row_start, row_end, dtype=np.int32)

    for col_start in range(0, N, B):
        col_end = min(col_start + B, N)
        Xj = X[col_start:col_end]

        sims = Xi @ Xj.T

        mask = (row_ids >= col_start) & (row_ids < col_end)
        if np.any(mask):
            local_rows = np.where(mask)[0]
            local_cols = row_ids[mask] - col_start
            sims[local_rows, local_cols] = -np.inf

        col_indices = np.arange(col_start, col_end, dtype=np.int32)
        col_indices_matrix = np.broadcast_to(col_indices, sims.shape)

        candidatos_values = np.concatenate((top_values, sims), axis=1)
        candidatos_indices = np.concatenate((top_indices, col_indices_matrix), axis=1)

        pos = np.argpartition(candidatos_values, -K, axis=1)[:, -K:]

        top_values = np.take_along_axis(candidatos_values, pos, axis=1)
        top_indices = np.take_along_axis(candidatos_indices, pos, axis=1)

    orden = np.argsort(-top_values, axis=1)

    top_values = np.take_along_axis(top_values, orden, axis=1)
    top_indices = np.take_along_axis(top_indices, orden, axis=1)

    shm.close()

    return row_start, top_values, top_indices


def calcular_topk_paralelo(X, N, D, B, K, workers):
    shm = shared_memory.SharedMemory(create=True, size=X.nbytes)

    X_shared = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
    X_shared[:] = X[:]

    top_values_global = np.empty((N, K), dtype=np.float32)
    top_indices_global = np.empty((N, K), dtype=np.int32)

    rangos = dividir_rangos(N, workers)

    args = [
        (inicio, fin, N, D, B, K, shm.name)
        for inicio, fin in rangos
    ]

    inicio_tiempo = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        resultados = list(executor.map(worker_topk, args))

    tiempo = time.perf_counter() - inicio_tiempo

    for row_start, top_values, top_indices in resultados:
        row_end = row_start + top_values.shape[0]
        top_values_global[row_start:row_end] = top_values
        top_indices_global[row_start:row_end] = top_indices

    shm.close()
    shm.unlink()

    return top_values_global, top_indices_global, tiempo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])

    args = parser.parse_args()

    N = args.n
    D = args.d
    B = args.block
    K = args.k

    cpu_count = mp.cpu_count()
    workers_list = [w for w in args.workers if w <= cpu_count]

    print("=== Cálculo de similitud por bloques para embeddings ===")
    print(f"N: {N}")
    print(f"D: {D}")
    print(f"Block size: {B}")
    print(f"Top-k: {K}")
    print(f"CPU disponibles: {cpu_count}")
    print(f"Workers a evaluar: {workers_list}")

    estimar_memoria(N, D, B, K)

    print("\nGenerando dataset sintético normalizado...")
    X = generar_embeddings(N, D)
    print("Dataset generado.")

    resultados = []
    tiempo_base = None

    for workers in workers_list:
        print(f"\nEjecutando con {workers} worker(s)...")

        top_values, top_indices, tiempo = calcular_topk_paralelo(
            X=X,
            N=N,
            D=D,
            B=B,
            K=K,
            workers=workers
        )

        if tiempo_base is None:
            tiempo_base = tiempo

        speedup = tiempo_base / tiempo
        eficiencia = speedup / workers
        comparaciones = N * (N - 1)
        throughput = comparaciones / tiempo

        resultados.append((workers, tiempo, speedup, eficiencia, throughput))

        print(f"Tiempo: {tiempo:.4f} s")
        print(f"Speedup: {speedup:.4f}")
        print(f"Eficiencia: {eficiencia:.4f}")
        print(f"Throughput: {throughput:.2f} comparaciones/s")

        print("\nTop-10 del vector 0:")
        for i in range(K):
            print(
                f"{i + 1:02d}) índice={top_indices[0, i]:6d} "
                f"similitud={top_values[0, i]:.6f}"
            )

    print("\n=== Tabla benchmark ===")
    print(f"{'Workers':>10} {'Tiempo(s)':>15} {'Speedup':>15} {'Eficiencia':>15} {'Throughput comp/s':>22}")

    for workers, tiempo, speedup, eficiencia, throughput in resultados:
        print(
            f"{workers:>10} "
            f"{tiempo:>15.4f} "
            f"{speedup:>15.4f} "
            f"{eficiencia:>15.4f} "
            f"{throughput:>22.2f}"
        )

    print("\nFórmulas:")
    print("Speedup S_p = T_1 / T_p")
    print("Eficiencia E_p = S_p / p")
    print("Throughput = N * (N - 1) / T_p")


if __name__ == "__main__":
    main()