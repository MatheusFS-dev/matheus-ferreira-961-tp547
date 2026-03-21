import random

p = 0.3
n_amostras = 100000


def geometric_sample(p):
    falhas = 0
    while True:
        if random.random() < p:
            return falhas
        falhas += 1


amostras = [geometric_sample(p) for _ in range(n_amostras)]

pb = (1 - p) ** 3 * p
pc = (1 - p) ** 6
media = (1 - p) / p
variancia = (1 - p) / (p**2)

media_simulada = sum(amostras) / n_amostras
variancia_simulada = sum((x - media_simulada) ** 2 for x in amostras) / n_amostras

print("Exercício 4")
print(f"Simulando {n_amostras} amostras da distribuição geométrica com p = {p}.")
print(f"   Média simulada = {media_simulada:.4f}")
print(f"   Variância simulada = {variancia_simulada:.4f}")
print(f"b) A probabilidade de o primeiro sucesso ocorrer na 4ª tentativa é {pb:.6f}, ou {pb * 100:.2f}%.")
print(f"c) A probabilidade de precisar de mais de 6 tentativas é {pc:.6f}, ou {pc * 100:.2f}%.")
print(f"d) A média é {media:.4f} e a variância é {variancia:.4f}.")
