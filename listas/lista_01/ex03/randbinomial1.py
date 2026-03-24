import numpy as np

# Probability of success (or event) in a single trial
q = 0.08
# Number of independent trials (n parameter for Binomial)
n = 20
# Value we want to find the exact probability for later (e.g., exactly 3 successes)
value = 3
# Number of samples to simulate to construct our distribution
N = 100000

# Constant factor used for the recurrence relation of the Binomial PMF: q / (1 - q)
c = q / (1 - q)
# Initialize an empty numpy array to store the generated Binomial samples
av = np.array([])
# Counter for occurrences of exactly 'value' (3)
count = 0
# Generate N random numbers from a Uniform(0,1) distribution
x = np.random.uniform(0, 1, N)

# Loop over each uniform random number to apply the inverse transform method
for ix in x:
    # Initialize the success count 'i' to 0
    i = 0
    # Calculate the PMF for i=0: P(X=0) = (1 - q)^n
    pr = pow((1 - q), n)
    # Initialize the Cumulative Distribution Function (CDF), F, with P(X=0)
    F = pr
    
    # While the uniform random number 'ix' is greater than or equal to the cumulative probability 'F'
    while ix >= F:
        # Compute the PMF for the next 'i' using the recurrence relation: P(X=i+1) = (c * (n - i) / (i + 1)) * P(X=i)
        pr = (c * (n - i) / (i + 1)) * pr
        # Add this new probability to the cumulative distribution 'F'
        F = F + pr
        # Increment the success count 'i'
        i = i + 1
        
    # The generated sample 'a1' is the value where the loop stopped
    a1 = i
    # Append the generated sample to the array
    av = np.append(av, a1)

# Print the array of the 100000 generated Binomial samples
print(av)

# Iterate through every generated sample in the array
for binvalue in av:
    # If the simulated sample matches the target 'value' (3)
    if binvalue == value:
        # Increment the counter 
        count = count + 1

# Calculate the simulated probability as the proportion of times the target value occurred
prob = count / N
# Print the calculated simulated probability for X = 3
print("a probabilidade e", prob)
