import math

# lambda for 1 hour (average number of events in 1 hour)
lamb_1h = 4
# lambda for 2 hours (average number of events in 2 hours, 4 * 2)
lamb_2h = 8


# Function to calculate the Probability Mass Function (PMF) of a Poisson distribution
# PMF gives the probability of exactly k events happening in a fixed interval
# Formula: P(X=k) = (e^-lambda * lambda^k) / k!
def poisson_pmf(k, lamb):
    # Calculate and return the PMF using the formula
    return (math.exp(-lamb) * (lamb**k)) / math.factorial(k)


# Function to calculate the Cumulative Distribution Function (CDF) of a Poisson distribution
# CDF gives the probability of at most k events happening
# It simply sums the PMF from 0 to k
def poisson_cdf(k, lamb):
    # Initialize the total probability sum
    total = 0
    # Loop from 0 up to k (inclusive)
    for i in range(k + 1):
        # Accumulate the PMF for each value
        total += poisson_pmf(i, lamb)
    # Return the accumulated probability
    return total


# a) Probability of exactly 6 failures in 1 hour
pa = poisson_pmf(6, lamb_1h)
# b) Probability of at most 2 failures in 1 hour (Cumulative probability from 0 to 2)
pb = poisson_cdf(2, lamb_1h)
# c) Probability of more than 5 failures in 2 hours
# Calculated as 1 minus the probability of at most 5 failures (complementary probability)
pc = 1 - poisson_cdf(5, lamb_2h)

# Expected value (mean) of a Poisson distribution is simply its lambda parameter
valor_esperado = lamb_1h
# Variance of a Poisson distribution is also its lambda parameter
variancia = lamb_1h

# Print the title for Exercise 2
print("Exercício 2")
# Print the formatted result for part a (both decimal and percentage format)
print(f"a) A probabilidade de ocorrerem exatamente 6 falhas em uma hora é {pa:.6f}, ou {pa * 100:.2f}%.")
# Print the formatted result for part b (both decimal and percentage format)
print(f"b) A probabilidade de ocorrerem no máximo 2 falhas em uma hora é {pb:.6f}, ou {pb * 100:.2f}%.")
# Print the formatted result for part c (both decimal and percentage format)
print(f"c) A probabilidade de ocorrerem mais de 5 falhas em duas horas é {pc:.6f}, ou {pc * 100:.2f}%.")
# Print the expected value and variance
print(f"d) O valor esperado é {valor_esperado} e a variância é {variancia}.")
