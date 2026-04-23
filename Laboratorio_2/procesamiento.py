import random
import math

def procesar_lote_datos(config: dict) -> dict:
    random.seed(config['semilla'])
    n = config['tamanio']
    datos = [random.gauss(mu=50, sigma=15) for _ in range(n)]
    mu = sum(datos) / n
    sigma = math.sqrt(sum((x - mu) ** 2 for x in datos) / n)
    normalizados = [(x - mu) / sigma for x in datos]
    return {
        'lote': config['semilla'],
        'media': mu,
        'std': sigma,
        'min_norm': min(normalizados),
        'max_norm': max(normalizados),
    }