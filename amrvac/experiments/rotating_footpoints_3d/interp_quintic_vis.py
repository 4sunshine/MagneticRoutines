import numpy as np
import matplotlib.pyplot as plt

def F_scalar(x, a, b):
    if x <= a:
        return 1.0
    elif x >= b:
        return 0.0
    else:
        u = (x - a) / (b - a)
        s = 6*u**5 - 15*u**4 + 10*u**3
        return 1.0 - s

F = np.vectorize(F_scalar)

# Choose interval [a, b]
a = 0.5
b = 4.0

xs = np.linspace(a - 1, b + 1, 800)
ys = F(xs, a, b)

plt.plot(xs, ys, linewidth=2)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("F(x)")
plt.title(f"Smoothstep transition from 1 to 0 on [{a}, {b}]")
plt.ylim(-0.1, 1.1)
plt.savefig(f"b_{b}_a_{a}.png")
