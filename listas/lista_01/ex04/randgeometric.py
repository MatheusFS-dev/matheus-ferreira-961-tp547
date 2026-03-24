import numpy as np
import matplotlib.pyplot as plt

# Parameter for the geometric distribution (Probability of success)
p = 0.3
# Number of samples to simulate to approximate the theoretical values
N = 100000
# Target value for calculating a specific frequency (X = number of failures before the first success)
value = 3

# Set the random seed for exact reproducibility
np.random.seed(42)

# Generate samples using the CDF inversion method
# Initialize an empty numpy array (type int) to store the simulated samples
av = np.array([], dtype=int)
# Generate N uniform random numbers between 0 and 1
x = np.random.uniform(0, 1, N)

# Loop over each uniform random number to find the corresponding geometric sample
for ix in x:
    # Initialize the failures counter `i` to 0
    i = 0
    # Calculate the PMF for exactly 0 failures: P(X=0) = p
    pr = p
    # Initialize the Cumulative Distribution Function (CDF), F, with P(X=0)
    F = pr
    
    # Check if the uniformly generated value is >= current CDF
    while ix >= F:
        # If so, increment the failure counter
        i = i + 1
        # Calculate the PMF for the next `i` using recurrence: P(X=i+1) = P(X=i) * (1-p)
        pr = (1 - p) * pr
        # Accumulate the new PMF into the CDF `F`
        F = F + pr
        
    # Append the number of failures `i` to the samples array
    av = np.append(av, i)

# Calculate simulated probability of exactly 3 failures (P(X=3)) using array boolean comparison
pb_sim = np.mean(av == value)
# Calculate simulated probability of 6 or more failures (P(T>=6), though printed as P(T>6) theoretically)
pc_sim = np.mean(av >= 6)
# Calculate the empirical mean of the simulated samples
media_sim = np.mean(av)
# Calculate the empirical variance of the simulated samples
var_sim = np.var(av)

# Print the simulated probability for X = 3 formatted to 6 decimal places
print(f'P(X = 3) simulada = {pb_sim:.6f}')
# Print the simulated probability for T > 6 formatted to 6 decimal places
print(f'P(T > 6) simulada = {pc_sim:.6f}')
# Print the simulated Mean formatted to 4 decimal places
print(f'Media simulada = {media_sim:.4f}')
# Print the simulated Variance formatted to 4 decimal places
print(f'Variancia simulada = {var_sim:.4f}')

# Retrieve unique values in the array and their corresponding frequency counts
k, counts = np.unique(av, return_counts=True)
# Calculate relative frequencies for plotting the empirical PMF
freq_simulada = counts / N

# Create a matplotlib figure object with size 10x5 inches
plt.figure(figsize=(10, 5))
# Plot a bar chart with the unique values `k` on x-axis and their relative frequencies on y-axis
plt.bar(k, freq_simulada, width=0.8, label='Simulada', alpha=0.85)
# Set the horizontal axis label
plt.xlabel('k (falhas antes do primeiro sucesso)')
# Set the vertical axis label
plt.ylabel('Frequencia relativa')
# Set the main plot title
plt.title('Distribuicao Geometrica Simulada')
# Show horizontal grid lines for scaling, with low opacity (0.3)
plt.grid(axis='y', alpha=0.3)
# Show the legend based on the labels specified above
plt.legend()
# Render and show the final plot
plt.show()
