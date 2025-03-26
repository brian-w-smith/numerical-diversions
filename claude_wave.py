import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create a figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set up the domain
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Set up the animation parameters
frames = 100
t_max = 10
t_values = np.linspace(0, t_max, frames)

# Initialize the surface plot
surf = [ax.plot_surface(X, Y, np.zeros_like(X), cmap=cm.plasma, alpha=0.8)]

# Function for updating the animation
def update(frame):
    # Clear the previous surface
    surf[0].remove()
    
    # Get the current time value
    t = t_values[frame]
    
    # Calculate the z-coordinate using a complex mathematical function
    # This combines several mathematical concepts:
    # - Sine waves with varying frequencies
    # - Damped oscillations based on the distance from the origin
    # - Time-varying phase shifts
    
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # First term: Modulated sine wave
    Z1 = 2 * np.sin(R - t) * np.exp(-0.1 * R)
    
    # Second term: Ripple effect with time-varying phase
    Z2 = 0.5 * np.sin(3 * R - 2 * t) * np.cos(theta)
    
    # Third term: Rotating spiral pattern
    Z3 = 0.3 * np.sin(5 * theta + t) * np.exp(-0.2 * R)
    
    # Fourth term: Bessel-like function
    Z4 = np.cos(R) * np.sin(t) * np.exp(-0.05 * R**2)
    
    # Combine all terms
    Z = Z1 + Z2 + Z3 + Z4
    
    # Update the plot with the new surface
    surf[0] = ax.plot_surface(X, Y, Z, cmap=cm.plasma, alpha=0.8)

    return surf

# Set up the axis labels and limits
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_zlim(-3, 3)

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

# Display the animation
plt.tight_layout()
plt.show()

# Uncomment the following line to save the animation
ani.save('claude_wave.mp4', writer='ffmpeg', fps=15, dpi=200)