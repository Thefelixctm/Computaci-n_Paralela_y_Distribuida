# calculos.py
def contar_primos(n):
    if n < 2: return 0
    criba = bytearray(b"\x01") * (n + 1)
    criba[0] = criba[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if criba[i]:
            criba[i * i : n + 1 : i] = bytearray(len(criba[i * i : n + 1 : i]))
    return int(sum(criba))