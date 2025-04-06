import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def solve_ode(A, B, C, t):
    """Solves the second-order linear homogeneous ODE."""
    discriminant = B**2 - 4*A*C

    if discriminant > 0:  # Two distinct real roots
        r1 = (-B + np.sqrt(discriminant)) / (2*A)
        r2 = (-B - np.sqrt(discriminant)) / (2*A)
        y = np.exp(r1*t) + np.exp(r2*t)
    elif discriminant == 0:  # One repeated real root
        r = -B / (2*A)
        y = np.exp(r*t) * (1 + t)
    else:  # Complex conjugate roots
        alpha = -B / (2*A)
        beta = np.sqrt(-discriminant) / (2*A)
        y = np.exp(alpha*t) * np.cos(beta*t) + np.exp(alpha*t) * np.sin(beta*t)

    return y

# Initial values
A_init = 1.0
B_init = 2.0
C_init = 1.0

# Time range
t = np.linspace(0, 10, 500)

# Create the figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# Plot the initial solution
line, = ax.plot(t, solve_ode(A_init, B_init, C_init, t), lw=2)

# Add sliders for A, B, and C
ax_A = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_B = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_C = plt.axes([0.25, 0.05, 0.65, 0.03])

slider_A = Slider(ax_A, 'A', 0.1, 5.0, valinit=A_init)
slider_B = Slider(ax_B, 'B', -5.0, 5.0, valinit=B_init)
slider_C = Slider(ax_C, 'C', -5.0, 5.0, valinit=C_init)

# Update the plot when sliders are changed
def update(val):
    A = slider_A.val
    B = slider_B.val
    C = slider_C.val
    line.set_ydata(solve_ode(A, B, C, t))
    fig.canvas.draw_idle()

slider_A.on_changed(update)
slider_B.on_changed(update)
slider_C.on_changed(update)

plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Solution of Ay'' + By' + Cy = 0")
plt.grid(True)
plt.show()