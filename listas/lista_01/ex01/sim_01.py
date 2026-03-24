import math

# a = multiplier for the LCG (Linear Congruential Generator)
a = 11
# c = increment for the LCG
c = 7
# m = modulus for the LCG (determines the max possible period)
m = 32
# x0 = seed (initial value) for the LCG
x0 = 5


# Function to generate the next pseudo-random number using the LCG formula
# The LCG formula is: X_{n+1} = (a * X_n + c) mod m
# Example: if a=11, x=5, c=7, m=32 -> (11*5 + 7) % 32 = (55 + 7) % 32 = 62 % 32 = 30
def next_lcg(x):
    # Calculate the next value by applying the linear congruential formula and returning the modulo m
    return (a * x + c) % m


# Initialize x with the seed value
x = x0
# Initialize an empty list to store the sequence of generated numbers
seq = []
# Loop 10 times to generate the first 10 numbers in the sequence
for _ in range(10):
    # Generate the next number in the sequence based on the current x
    x = next_lcg(x)
    # Append the newly generated number to the sequence list
    seq.append(x)

# Print the label for part a)
print("a)")
# Print the generated sequence of 10 numbers
print(seq)

# Initialize a dictionary to keep track of seen numbers and their indices to detect cycles
seen = {}
# Initialize a list to store the entire orbit (all generated numbers until a repetition occurs)
orbit = []
# Re-initialize x with the seed value to start fresh
x = x0

# Loop indefinitely until a previously generated number is encountered (which means a cycle started)
# This is a common logic to find the period of a sequence. It stores each state and checks if it was already visited.
# Example: if x=5 was seen at index 0, and we see x=5 again at index 10, the period is 10 - 0 = 10.
while x not in seen:
    # Store the current number in the dictionary with its index in the orbit
    seen[x] = len(orbit)
    # Append the current number to the orbit list
    orbit.append(x)
    # Generate the next number for the next iteration
    x = next_lcg(x)

# Calculate the period of the generator by subtracting the index where the repeated number was first seen from the total length of the orbit
periodo = len(orbit) - seen[x]

# Print an empty line for better formatting
print()
# Print the label and the calculated period for part b)
print(f"b) {periodo}.")

# Check the first condition for maximum period: the greatest common divisor of c and m must be 1 (they are coprime)
cond1 = math.gcd(c, m) == 1

# Initialize a set to store the prime factors of m
primos = set()
# Create a temporary variable to hold the value of m for factorization
temp = m
# Start with the smallest prime number for trial division
d = 2
# Loop to find all prime factors of m by trial division up to the square root of m
# This algorithm divides `temp` by `d` as many times as possible, then increments `d`.
# Example: temp = 32. d = 2. 32%2==0 -> primos.add(2), temp=16. 16%2==0 -> temp=8... until temp=1.
while d * d <= temp:
    # While d evenly divides temp, it is a prime factor
    while temp % d == 0:
        # Add the prime factor to the set
        primos.add(d)
        # Divide temp by d to remove this factor from temp
        temp //= d
    # Increment d to test the next potential factor
    d += 1
# If temp is greater than 1 after the previous loop, it means temp itself is a prime factor
if temp > 1:
    # Add the remaining prime factor to the set
    primos.add(temp)

# Check the second condition for maximum period: a - 1 must be a multiple of every prime factor of m
# It evaluates to True if (a - 1) % p == 0 for all p in the `primos` set.
cond2 = all((a - 1) % p == 0 for p in primos)
# Check the third condition: if m is a multiple of 4, a - 1 must also be a multiple of 4
cond3 = (m % 4 != 0) or ((a - 1) % 4 == 0)

# Print an empty line for better formatting
print()
# Print the label for part c)
print("c)")
# Print the boolean result of the first condition
print(f"- mdc(c, m) = 1: {cond1}")
# Print the boolean result of the second condition
print(f"- Todo primo divisor de m divide (a - 1): {cond2}")
# Print the boolean result of the third condition
print(f"- Se m é múltiplo de 4, então (a - 1) também deve ser: {cond3}")

# If all three conditions are satisfied, the generator has the maximum possible period
if cond1 and cond2 and cond3:
    # Print that the generator satisfies the conditions for maximum period
    print(f"Logo, o gerador satisfaz o período máximo, que seria {m}.")
# Otherwise, it does not have the maximum period
else:
    # Print that it does not satisfy the conditions and show the actual period versus the maximum
    print(
        f"Logo, o gerador não satisfaz o período máximo. O período obtido foi {periodo}, mas o máximo seria {m}."
    )
