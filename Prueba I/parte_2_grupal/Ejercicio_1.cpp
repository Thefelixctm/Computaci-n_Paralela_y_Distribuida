#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCOLS 16

static const char *COL_NAMES[NCOLS] = {
    "pickup_latitude",  "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude",
    "trip_distance_km", "fare_amount_usd",
    "tip_amount_usd",   "tolls_amount_usd",
    "total_amount_usd", "avg_speed_kmh",
    "max_speed_kmh",    "traffic_time_min",
    "trip_duration_min","surge_multiplier",
    "passenger_count",  "driver_rating"
};

int main(int argc, char **argv) {
    const char *filename = "dataset_matriz_X.bin";
    long nrows = 50000000L;

    if (argc > 1) filename = argv[1];
    if (argc > 2) nrows = atol(argv[2]);

    long nelem = nrows * (long)NCOLS;
    size_t nbytes = (size_t)nelem * sizeof(double);

    printf("============================================\n");
    printf("  Normalizacion Z-Score con OpenMP\n");
    printf("============================================\n");
    printf("  Dataset: %s\n", filename);
    printf("  Filas:   %ld\n", nrows);
    printf("  Columnas:%d\n", NCOLS);
    printf("  Tamanio: %.2f GB\n", nbytes / 1e9);
    printf("--------------------------------------------\n");
    printf("  CPUs disponibles: %d\n", omp_get_num_procs());
    printf("  Max hilos OpenMP: %d\n", omp_get_max_threads());
    printf("--------------------------------------------\n");

    /* --- Allocate --- */
    double *data = (double *)malloc(nbytes);
    if (!data) {
        fprintf(stderr, "ERROR: No se pudo asignar %.2f GB de memoria.\n",
                nbytes / 1e9);
        return 1;
    }

    /* --- Read binary file --- */
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: No se pudo abrir %s\n", filename);
        free(data);
        return 1;
    }
    size_t read_bytes = fread(data, 1, nbytes, f);
    fclose(f);
    if (read_bytes != nbytes) {
        fprintf(stderr, "ERROR: Se leyeron %zu bytes, esperados %zu\n",
                read_bytes, nbytes);
        free(data);
        return 1;
    }
    printf("  Archivo leido correctamente.\n");
    printf("--------------------------------------------\n");

    /* ================================================
       FASE 1: Media (mu) y conteo valido por columna
       ================================================ */
    double t1_start = omp_get_wtime();

    double sums[NCOLS];
    long   vals[NCOLS];
    for (int j = 0; j < NCOLS; j++) { sums[j] = 0.0; vals[j] = 0; }

    #pragma omp parallel
    {
        double local_sums[NCOLS];
        long   local_vals[NCOLS];
        for (int j = 0; j < NCOLS; j++) { local_sums[j] = 0.0; local_vals[j] = 0; }

        #pragma omp for nowait
        for (long i = 0; i < nrows; i++) {
            long base = i * NCOLS;
            for (int j = 0; j < NCOLS; j++) {
                double v = data[base + j];
                if (!isnan(v)) {
                    local_sums[j] += v;
                    local_vals[j]++;
                }
            }
        }

        #pragma omp critical
        {
            for (int j = 0; j < NCOLS; j++) {
                sums[j] += local_sums[j];
                vals[j] += local_vals[j];
            }
        }
    }

    double mu[NCOLS];
    long total_valid = 0;
    for (int j = 0; j < NCOLS; j++) {
        mu[j] = sums[j] / vals[j];
        total_valid += vals[j];
    }
    long total_nan = nelem - total_valid;

    double t1_end = omp_get_wtime();
    double t_fase1 = t1_end - t1_start;
    printf("  FASE 1 - Media por columna:\n");
    for (int j = 0; j < NCOLS; j++) {
        printf("    %-20s  mu = %12.4f  (valores=%ld)\n",
               COL_NAMES[j], mu[j], vals[j]);
    }
    printf("  NaN total: %ld (%.2f%%)\n", total_nan,
           100.0 * total_nan / nelem);
    printf("  Tiempo Fase 1 (media): %.4f s\n", t_fase1);
    printf("--------------------------------------------\n");

    /* ================================================
       FASE 2: Varianza (sigma^2) por columna
       ================================================ */
    double t2_start = omp_get_wtime();

    double sum_sq[NCOLS];
    for (int j = 0; j < NCOLS; j++) sum_sq[j] = 0.0;

    #pragma omp parallel
    {
        double local_sum_sq[NCOLS];
        for (int j = 0; j < NCOLS; j++) local_sum_sq[j] = 0.0;

        #pragma omp for nowait
        for (long i = 0; i < nrows; i++) {
            long base = i * NCOLS;
            for (int j = 0; j < NCOLS; j++) {
                double v = data[base + j];
                if (!isnan(v)) {
                    double diff = v - mu[j];
                    local_sum_sq[j] += diff * diff;
                }
            }
        }

        #pragma omp critical
        {
            for (int j = 0; j < NCOLS; j++)
                sum_sq[j] += local_sum_sq[j];
        }
    }

    double sigma[NCOLS];
    for (int j = 0; j < NCOLS; j++)
        sigma[j] = sqrt(sum_sq[j] / vals[j]);

    double t2_end = omp_get_wtime();
    double t_fase2 = t2_end - t2_start;
    printf("  FASE 2 - Desviacion estandar por columna:\n");
    for (int j = 0; j < NCOLS; j++) {
        printf("    %-20s  sigma = %12.4f\n",
               COL_NAMES[j], sigma[j]);
    }
    printf("  Tiempo Fase 2 (varianza): %.4f s\n", t_fase2);
    printf("--------------------------------------------\n");

    /* ================================================
       FASE 3: Normalizacion Z-Score y conteo de atipicos
       ================================================ */
    double t3_start = omp_get_wtime();

    long outliers = 0;

    #pragma omp parallel for reduction(+:outliers)
    for (long i = 0; i < nrows; i++) {
        long base = i * NCOLS;
        for (int j = 0; j < NCOLS; j++) {
            double v = data[base + j];
            if (!isnan(v)) {
                double z = (v - mu[j]) / sigma[j];
                data[base + j] = z;
                if (fabs(z) > 3.0) outliers++;
            }
        }
    }

    double t3_end = omp_get_wtime();
    double t_fase3 = t3_end - t3_start;
    printf("  FASE 3 - Normalizacion Z-Score completada.\n");
    printf("  Atipicos (|Z|>3): %ld (%.2f%%)\n", outliers,
           100.0 * outliers / total_valid);
    printf("  Tiempo Fase 3 (norm+outliers): %.4f s\n", t_fase3);
    printf("--------------------------------------------\n");

    double t_total = t_fase1 + t_fase2 + t_fase3;
    printf("  Tiempo total (fase1+fase2+fase3): %.4f s\n", t_total);
    printf("--------------------------------------------\n");

    /* --- Muestra --- */
    printf("  Muestra (primeras 3 filas normalizadas Z-Score):\n");
    for (long i = 0; i < 3 && i < nrows; i++) {
        printf("    fila %ld:", i);
        for (int j = 0; j < NCOLS; j++) {
            printf(" %8.4f", data[i * NCOLS + j]);
        }
        printf("\n");
    }
    printf("============================================\n");

    free(data);
    return 0;
}
