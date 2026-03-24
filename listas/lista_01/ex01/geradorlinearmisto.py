import numpy as np
import matplotlib.pyplot as plt

# Initial seed value for the pseudo-random number generator
x = 5
# Create a numpy array starting with the initial seed value
x1 = np.array([x])
# Number of random values to generate
n = 100
# Multiplier parameter (a) for the Linear Congruential Generator (LCG)
a = 11
# Increment parameter (c) for the LCG
c = 7
# Modulus parameter (m) for the LCG
m = 32

# Loop n times to iteratively generate points
for i in range(n):
    # Apply the LCG formula: X_{n+1} = (a * X_n + c) mod m
    # This creates a pseudo-random sequence of numbers between 0 and m-1
    x = (a * x + c) % m
    # Append the newly generated number to our sequence array x1
    x1 = np.append(x1, x)

# Print the final array containing all generated values
print(x1)

# Create an array of indices from 0 to n to act as the x-axis for plotting
ind = np.arange(n + 1)
# Plot a bar chart with the index on the x-axis and the generated values on the y-axis
plt.bar(ind, x1)
# Display the chart
plt.show()
