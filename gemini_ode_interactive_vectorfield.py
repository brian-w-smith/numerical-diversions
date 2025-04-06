import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def vector_field(y, y_prime, A, B, C):
    """Calculates the vector field for the ODE."""
    y_double_prime = (-B * y_prime - C * y) / A
    return y_prime, y_double_prime

# Initial values
A_init = 1.0
B_init = 2.0
C_init = 1.0

# Range for y and y_prime
y_range = np.linspace(-5, 5, 20)
y_prime_range = np.linspace(-5, 5, 20)
Y, Y_prime = np.meshgrid(y_range, y_prime_range)

# Create the figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# Calculate and plot the initial vector field
U, V = vector_field(Y, Y_prime, A_init, B_init, C_init)
quiver = ax.quiver(Y, Y_prime, U, V, color='b')

# Add sliders for A, B, and C
ax_A = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_B = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_C = plt.axes([0.25, 0.05, 0.65, 0.03])

slider_A = Slider(ax_A, 'A', 0.1, 5.0, valinit=A_init)
slider_B = Slider(ax_B, 'B', -5.0, 5.0, valinit=B_init)
slider_C = Slider(ax_C, 'C', -5.0, 5.0, valinit=C_init)

# Update the vector field when sliders are changed
def update(val):
    A = slider_A.val
    B = slider_B.val
    C = slider_C.val
    U, V = vector_field(Y, Y_prime, A, B, C)
    quiver.set_UVC(U, V)  # Update the quiver plot's U and V data
    fig.canvas.draw_idle()

slider_A.on_changed(update)
slider_B.on_changed(update)
slider_C.on_changed(update)

plt.xlabel('y')
plt.ylabel('y\'')
plt.title("Vector Field of Ay'' + By' + Cy = 0")
plt.grid(True)
plt.show()