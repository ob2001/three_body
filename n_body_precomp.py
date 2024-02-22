from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 8.9953e-6 # Adjusted for normalized units
n_bodies = 3
steps = 4000
dt = 0.25
scale = 0.5

bodies = np.zeros([steps, n_bodies, 5])

# Masses are measured in Earth masses
# Distances are measured in AU
# Time is measured in 86400s
# Near approximation to Earth-Sun system
# The data for each body is in the order [m, x, y, v_x, v_y]
bodies[0] = np.array([
                np.array([100., 0., 0., 0., 0.]),
                np.array([1., 0., 1., 0.027, 0.]),
                np.array([1., 0., -0.5, -0.027, 0.])
            ])

def mot_eq(b, t, i):
    m, p, v = b[0], b[1:3], b[3::]
    f = np.zeros([2])
    for j in range(len(bodies[t])):
        if j != i:
            f += (bodies[t][j][1:3] - p)*G*m*bodies[t][j][0]/np.linalg.norm(bodies[t][j][1:3] - p)**3
    return np.array([0., v[0], v[1], f[0]/m, f[1]/m])

def rk4(dt, f, v, t, i):
    k1 = dt*f(v, t, i)
    k2 = dt*f(v + k1/2, t, i)
    k3 = dt*f(v + k2/2, t, i)
    k4 = dt*f(v + k3, t, i)
    return v + (k1 + 2*k2 + 2*k3 + k4)/6

def anim(t, ax, bodies):
    ax.clear()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"$E={Es[t]:.5}$")
    for body in bodies[t]:
        ax.plot(body[1], body[2], 'o')
        ax.quiver(body[1], body[2], body[3], body[4], scale = scale, width = 0.004)
    return

# def grav_force(b1, b2):
#     return (b2[1:3] - b1[1:3])*G*b1[0]/np.linalg.norm(b2[1:3] - b1[1:3])**3

def grav_energy(b1, b2):
    return -G*b1[0]*b2[0]/np.linalg.norm(b2[1:3] - b1[1:3])

for t in range(steps - 1):
    next = []
    for i in range(len(bodies[t])):
        next.append(rk4(dt, mot_eq, bodies[t][i], t, i))
    for i in range(len(bodies[t])):
        bodies[t + 1] = np.array(next)

Ks = np.array([np.sum([0.5*b[0]*np.linalg.norm(b[3::])**2 for b in frame]) for frame in bodies])
Vs = np.array([np.sum([grav_energy(b1, b2) for b1, b2 in combinations(frame, 2)]) for frame in bodies])
Es = Ks + Vs

# for (K, V, E) in zip(Ks, Vs, Es): print(K, V, E)

fig = plt.figure(figsize = (8, 7))
ax = fig.add_subplot(111, xlim = (-1.1, 1.1), ylim = (-1.1, 1.1), title = f"$E={Es[0]:.5}$")

for body in bodies[0]:
    ax.plot(body[1], body[2], 'o')
    ax.quiver(body[1], body[2], body[3], body[4], scale = scale, width = 0.004)

ani = FuncAnimation(fig, anim, frames = steps, repeat = True, fargs = (ax, bodies), interval = 50)

plt.show()