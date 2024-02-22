from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def mot_eq(b, i):
    m, p, v = b[0], b[1:3], b[3::]
    f = np.zeros([2])
    for j in range(len(bodies)):
        if j != i:
            f += (bodies[j][1:3] - p)*G*m*bodies[j][0]/np.linalg.norm(bodies[j][1:3] - p)**3
    return np.array([0., *v, *f/m])

def rk4(dt, f, v, i):
    k1 = dt*f(v, i)
    k2 = dt*f(v + k1/2, i)
    k3 = dt*f(v + k2/2, i)
    k4 = dt*f(v + k3, i)
    return v + (k1 + 2*k2 + 2*k3 + k4)/6

def anim(t, ax, plot, quiver):
    global bodies
    temp_bodies = []
    for i in range(len(bodies)):
        temp_bodies.append(rk4(dt, mot_eq, bodies[i], i))
    bodies = np.array(temp_bodies)

    ax.set_title(f"E: {energy(bodies):.5}") # Won't work if blitting
    plot.set_data(bodies[:,1], bodies[:,2])    
    quiver.set_offsets(np.c_[bodies[:,1], bodies[:,2]])
    quiver.set_UVC(bodies[:,3], bodies[:,4])

    return (plot, quiver)

def grav_energy(b1, b2):
    return -G*b1[0]*b2[0]/np.linalg.norm(b2[1:3] - b1[1:3])

def energy(bodies):
    K = np.sum([0.5*b[0]*np.linalg.norm(b[3::])**2 for b in bodies])
    V = np.sum([grav_energy(b1, b2) for b1, b2 in combinations(bodies, 2)])
    return K + V

""" Constants """
G = 8.9953e-6 # Adjusted for normalized units
dt = 0.25
scale = 0.5

""" Simulation bodies """
# Masses are measured in Earth masses
# Distances are measured in AU
# Time is measured in 86400s
# Near approximation to Earth-Sun system
# The data for each body is in the order [m, x, y, v_x, v_y]
bodies = np.array([
                    np.array([100., 0., 0., 0., 0.]),
                    np.array([1., 0., 1., 0.027, 0.]),
                    np.array([1., 0., -0.5, -0.027, 0.])
                ])

""" Plotting """
fig = plt.figure(figsize = (8, 7))
ax = fig.add_subplot(111, xlim = (-1.1, 1.1), ylim = (-1.1, 1.1), title = f"E: {energy(bodies):.5}")

plot, = ax.plot(bodies[:,1], bodies[:,2], 'o')
quiver = ax.quiver(bodies[:,1], bodies[:,2], bodies[:,3], bodies[:,4], scale = scale, width = 0.004)

ani = FuncAnimation(fig, anim, frames = None, repeat = True, fargs = (ax, plot, quiver), interval = 10, save_count = 0, blit = False)

plt.show()