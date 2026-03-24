import numpy as np

# Average number of requests/events in the given interval (lambda)
lambda1 = 4
# Number of samples to generate in the simulation
N = 100

# Initialize an empty numpy array to store the generated Poisson samples
av = np.array([])
# Generate N random numbers from a Uniform(0,1) distribution
x = np.random.uniform(0, 1, N)

# Loop over each generated uniform random number to apply the inverse transform method
for ix in x:
    # Initialize the event count 'i' to 0 for the current sample
    i = 0
    # Calculate the PMF for i=0: P(X=0) = e^(-lambda)
    pr = np.exp(-lambda1)
    # Initialize the Cumulative Distribution Function (CDF), F, with P(X=0)
    F = pr
    
    # While the uniform random number 'ix' is greater than or equal to the cumulative probability 'F'
    while ix >= F:
        # Calculate the PMF for the next 'i' using the recurrence relation: P(X=i+1) = (lambda / (i+1)) * P(X=i)
        pr = lambda1 / (i + 1) * pr
        # Add the new probability to the cumulative distribution 'F'
        F = F + pr
        # Increment the event count 'i'
        i = i + 1
        
    # The generated sample 'a1' is the value of 'i' where ix < F
    a1 = i
    # Append the generated sample to our array of samples
    av = np.append(av, a1)

# Print the final array containing the 100 generated Poisson samples
print(av)
