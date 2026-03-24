import math

# Total number of trials (e.g., bits transmitted)
n = 20
# Probability of success (or error, in this context) in a single trial
p = 0.08


# Function to calculate the Probability Mass Function (PMF) of a Binomial distribution
# PMF gives the probability of exactly k successes in n independent trials
# Formula: P(X=k) = C(n, k) * p^k * (1-p)^(n-k)
# Example: choosing 3 errors out of 20 -> math.comb(20, 3) * 0.08^3 * 0.92^17
def binomial_pmf(k):
    # Calculate combinations C(n, k) and multiply by the probabilities
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))


# a) Probability of exactly 3 errors (successes in this context) out of 20
pa = binomial_pmf(3)
# b) Probability of at most 2 errors (sum of probabilities for k=0, 1, 2)
pb = sum(binomial_pmf(k) for k in range(3))
# c) Probability of more than 5 errors
# Calculated as 1 minus the cumulative probability of at most 5 errors (k=0 through 5)
pc = 1 - sum(binomial_pmf(k) for k in range(6))

# The mean (expected value) of a Binomial distribution is n * p
media = n * p
# The variance of a Binomial distribution is n * p * (1 - p)
variancia = n * p * (1 - p)

# Print the title for Exercise 3
print("Exercício 3")
# Print the formatted result for part a (both decimal and percentage format)
print(f"a) A probabilidade de ocorrerem exatamente 3 erros é {pa:.6f}, ou {pa * 100:.2f}%.")
# Print the formatted result for part b (both decimal and percentage format)
print(f"b) A probabilidade de ocorrerem no máximo 2 erros é {pb:.6f}, ou {pb * 100:.2f}%.")
# Print the formatted result for part c (both decimal and percentage format)
print(f"c) A probabilidade de ocorrerem mais de 5 erros é {pc:.6f}, ou {pc * 100:.2f}%.")
# Print the mean and variance rounded to 3 decimal places
print(f"d) A média é {media:.3f} e a variância é {variancia:.3f}.")
