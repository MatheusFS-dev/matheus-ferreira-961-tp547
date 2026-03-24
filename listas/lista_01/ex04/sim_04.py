import random

# Probability of success in a single trial
p = 0.3
# Number of samples to simulate to approximate the theoretical values
n_amostras = 100000


# Function to generate a single sample from a Geometric distribution
# The Geometric distribution models the number of failures before the first success
# Example: if random values are [0.5, 0.4, 0.2] and p=0.3, it fails twice and succeeds on the 3rd try (returns 2 failures)
def geometric_sample(p):
    # Initialize the counter for the number of failures
    falhas = 0
    # Loop indefinitely until a success occurs
    while True:
        # Generate a random float between 0.0 and 1.0, and check if it's less than probability p (indicating success)
        if random.random() < p:
            # If success, return the number of failures that occurred before it
            return falhas
        # If not a success, increment the failure counter
        falhas += 1


# Generate a list of samples by running the geometric_sample function n_amostras times
amostras = [geometric_sample(p) for _ in range(n_amostras)]

# b) Theoretical probability that the first success occurs exactly on the 4th trial
# It means 3 failures (1-p) followed by 1 success (p): (1 - p)^3 * p
pb = (1 - p) ** 3 * p
# c) Theoretical probability of needing MORE than 6 trials for the first success
# It means the first 6 trials must all be failures: (1 - p)^6
pc = (1 - p) ** 6
# d) Theoretical mean (expected value) of the number of failures before the first success
media = (1 - p) / p
# Theoretical variance of the geometric distribution (failures before first success)
variancia = (1 - p) / (p**2)

# Calculate the simulated mean by dividing the sum of all samples by the number of samples
media_simulada = sum(amostras) / n_amostras
# Calculate the simulated variance using the sample variance formula: sum((x - mean)^2) / n
variancia_simulada = sum((x - media_simulada) ** 2 for x in amostras) / n_amostras

# Print the title for Exercise 4
print("Exercício 4")
# Print a message indicating the number of samples and the probability parameter being simulated
print(f"Simulando {n_amostras} amostras da distribuição geométrica com p = {p}.")
# Print the calculated practical/simulated mean
print(f"   Média simulada = {media_simulada:.4f}")
# Print the calculated practical/simulated variance
print(f"   Variância simulada = {variancia_simulada:.4f}")
# Print the formatted result for part b (both decimal and percentage format)
print(f"b) A probabilidade de o primeiro sucesso ocorrer na 4ª tentativa é {pb:.6f}, ou {pb * 100:.2f}%.")
# Print the formatted result for part c (both decimal and percentage format)
print(f"c) A probabilidade de precisar de mais de 6 tentativas é {pc:.6f}, ou {pc * 100:.2f}%.")
# Print the theoretical mean and variance
print(f"d) A média é {media:.4f} e a variância é {variancia:.4f}.")
