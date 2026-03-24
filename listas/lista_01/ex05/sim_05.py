import numpy as np
import matplotlib.pyplot as plt

# Number of samples to simulate
n = 100000

# Inverse transform sampling method generator:
# Set the seed for reproducibility of random numbers
np.random.seed(42)

# Generate 'n' uniform random numbers between 0 and 1
# Since U ~ Uniform(0,1), it serves as the base for the inverse transform
u = np.random.uniform(0, 1, n)

# Apply the inverse cumulative distribution function (CDF) to uniform samples
# In this case, the theoretical PDF is f(x) = 2x, so CDF is F(x) = x^2. Thus, the inverse is X = sqrt(U).
# Example: if U represents cumulative probability 0.25, sqrt(0.25) = 0.5 is the sampled value.
amostras = np.sqrt(u)

# Print out the total number of samples generated
print(f"Foram geradas {n} amostras")
# Calculate and print the minimum value in the simulated samples array
print(f"Minimo: {amostras.min():.4f}")
# Calculate and print the maximum value in the simulated samples array
print(f"Maximo: {amostras.max():.4f}")
# Calculate and print the mean of the simulated samples
print(f"Media simulada: {amostras.mean():.4f}")
# Calculate and print the variance of the simulated samples
print(f"Variancia simulada: {amostras.var():.4f}")

# Setup for the theoretical PDF to plot over the histogram
# Generate 500 evenly spaced x values from 0 to 1
x_vals = np.linspace(0, 1, 500)
# Calculate the corresponding y values using the theoretical PDF formula f(x) = 2x
y_vals = 2 * x_vals

# Initialize a matplotlib figure with a specified size of 9x5 inches
plt.figure(figsize=(9, 5))
# Plot a normalized histogram (density=True) with 25 bins, 70% opacity, and a label
plt.hist(amostras, bins=25, density=True, alpha=0.7, label="Simulada")
# Plot the theoretical PDF as a red line with thickness 2, and add a label
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="PDF teorica: f(x) = 2x")
# Set the label for the x-axis
plt.xlabel("x")
# Set the label for the y-axis
plt.ylabel("Densidade")
# Set the title of the plot
plt.title("Histograma das amostras e PDF teorica")
# Enable horizontal grid lines with low opacity (30%)
plt.grid(axis="y", alpha=0.3)
# Show the legend to identify the histogram and theoretical line
plt.legend()
# Render and display the plot
plt.show()
