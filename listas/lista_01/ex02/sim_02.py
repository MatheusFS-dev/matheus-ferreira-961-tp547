import math

lamb_1h = 4
lamb_2h = 8


def poisson_pmf(k, lamb):
    return (math.exp(-lamb) * (lamb**k)) / math.factorial(k)


def poisson_cdf(k, lamb):
    total = 0
    for i in range(k + 1):
        total += poisson_pmf(i, lamb)
    return total


pa = poisson_pmf(6, lamb_1h)
pb = poisson_cdf(2, lamb_1h)
pc = 1 - poisson_cdf(5, lamb_2h)

valor_esperado = lamb_1h
variancia = lamb_1h

print("Exercício 2")
print(f"a) A probabilidade de ocorrerem exatamente 6 falhas em uma hora é {pa:.6f}, ou {pa * 100:.2f}%.")
print(f"b) A probabilidade de ocorrerem no máximo 2 falhas em uma hora é {pb:.6f}, ou {pb * 100:.2f}%.")
print(f"c) A probabilidade de ocorrerem mais de 5 falhas em duas horas é {pc:.6f}, ou {pc * 100:.2f}%.")
print(f"d) O valor esperado é {valor_esperado} e a variância é {variancia}.")
