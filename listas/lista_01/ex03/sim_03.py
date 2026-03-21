import math

n = 20
p = 0.08


def binomial_pmf(k):
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))


pa = binomial_pmf(3)
pb = sum(binomial_pmf(k) for k in range(3))
pc = 1 - sum(binomial_pmf(k) for k in range(6))
media = n * p
variancia = n * p * (1 - p)

print("Exercício 3")
print(f"a) A probabilidade de ocorrerem exatamente 3 erros é {pa:.6f}, ou {pa * 100:.2f}%.")
print(f"b) A probabilidade de ocorrerem no máximo 2 erros é {pb:.6f}, ou {pb * 100:.2f}%.")
print(f"c) A probabilidade de ocorrerem mais de 5 erros é {pc:.6f}, ou {pc * 100:.2f}%.")
print(f"d) A média é {media:.3f} e a variância é {variancia:.3f}.")
