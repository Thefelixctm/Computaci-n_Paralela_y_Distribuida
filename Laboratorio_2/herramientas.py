import math
# La función debe estar en un archivo físico para que Windows la vea
def integrar_trapecios(segmento):
    a, b, N = segmento
    h = (b - a) / N
    total = 0.0
    for k in range(N):
        x = a + k * h
        total += math.sin(x) * math.exp(-x) * h
    return total