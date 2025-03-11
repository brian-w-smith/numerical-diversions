import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Define the function to integrate
def f(x, y):
    return np.sin(x) * np.cos(y)

# Create a grid of x, y values
x = np.linspace(0, np.pi, 100)
y = np.linspace(0, np.pi, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Calculate the double integral using numerical methods
dx = x[1] - x[0]
dy = y[1] - y[0]
integral_value = np.sum(Z) * dx * dy

# Set up figure
fig = plt.figure(figsize=(15, 10))

# Plot 1: The function f(x,y)
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('Function f(x,y) = sin(x)cos(y)')

# Plot 2: Visualization of the double integral as volume under the surface
ax2 = fig.add_subplot(122, projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.3)

# Fill the volume under the surface to visualize the integral
for i in range(0, len(x), 10):
    for j in range(0, len(y), 10):
        # Add "columns" from the xy-plane to the surface
        ax2.plot([X[i,j], X[i,j]], [Y[i,j], Y[i,j]], [0, Z[i,j]], 'r-', alpha=0.2)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')
ax2.set_title(f'Volume under f(x,y) = ∫∫sin(x)cos(y)dxdy ≈ {integral_value:.4f}')

# Add a plane at z=0 to represent the xy-plane
xx, yy = np.meshgrid(np.linspace(0, np.pi, 2), np.linspace(0, np.pi, 2))
ax2.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.3)

plt.tight_layout()
plt.suptitle('Double Integral Visualization', fontsize=16)
plt.subplots_adjust(top=0.9)

# Show the result
plt.show()

# Analytical verification of the integral
exact_value = 2  # The exact value of ∫∫sin(x)cos(y)dxdy over [0,π]×[0,π]
print(f"Numerical approximation: {integral_value:.6f}")
print(f"Exact value: {exact_value}")
print(f"Relative error: {(abs(integral_value - exact_value)/exact_value)*100:.6f}%")