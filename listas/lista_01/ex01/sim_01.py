import math

a = 11
c = 7
m = 32
x0 = 5


def next_lcg(x):
    return (a * x + c) % m


x = x0
seq = []
for _ in range(10):
    x = next_lcg(x)
    seq.append(x)

print("a)")
print(seq)

seen = {}
orbit = []
x = x0

while x not in seen:
    seen[x] = len(orbit)
    orbit.append(x)
    x = next_lcg(x)

periodo = len(orbit) - seen[x]

print()
print(f"b) {periodo}.")

cond1 = math.gcd(c, m) == 1

primos = set()
temp = m
d = 2
while d * d <= temp:
    while temp % d == 0:
        primos.add(d)
        temp //= d
    d += 1
if temp > 1:
    primos.add(temp)

cond2 = all((a - 1) % p == 0 for p in primos)
cond3 = (m % 4 != 0) or ((a - 1) % 4 == 0)

print()
print("c)")
print(f"- mdc(c, m) = 1: {cond1}")
print(f"- Todo primo divisor de m divide (a - 1): {cond2}")
print(f"- Se m é múltiplo de 4, então (a - 1) também deve ser: {cond3}")

if cond1 and cond2 and cond3:
    print(f"Logo, o gerador satisfaz o período máximo, que seria {m}.")
else:
    print(
        f"Logo, o gerador não satisfaz o período máximo. O período obtido foi {periodo}, mas o máximo seria {m}."
    )
