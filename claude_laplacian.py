import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create a figure with subplots
fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Create meshgrid
x = np.linspace(-5, 5, 100)
t = np.linspace(0, 10, 100)  # time variable instead of y
X, T = np.meshgrid(x, t)

# Initial parameters
wave_speed = 1.0
frequency = 0.5
amplitude = 1.0

# Wave function (solution to the wave equation)
def wave_function(x, t, c, f, A):
    """
    A traveling wave solution of the wave equation
    c: wave speed
    f: frequency
    A: amplitude
    """
    k = 2 * np.pi * f / c  # wave number
    omega = 2 * np.pi * f   # angular frequency
    return A * np.sin(k * x - omega * t)

# Laplacian (spatial second derivative) of the wave function
def laplacian_wave(x, t, c, f, A):
    """
    The spatial second derivative (Laplacian in 1D) of the wave function
    For the wave equation: ∂²u/∂t² = c² * ∂²u/∂x²
    """
    k = 2 * np.pi * f / c  # wave number
    omega = 2 * np.pi * f   # angular frequency
    return -A * k**2 * np.sin(k * x - omega * t)  # ∂²u/∂x² = -k² * u

# Initial plots
Z_wave = wave_function(X, T, wave_speed, frequency, amplitude)
Z_laplacian = laplacian_wave(X, T, wave_speed, frequency, amplitude)

# Create initial surfaces
surf1 = ax1.plot_surface(X, T, Z_wave, cmap='viridis', alpha=0.8)
surf2 = ax2.plot_surface(X, T, Z_laplacian, cmap='coolwarm', alpha=0.8)

# Wave equation in text
wave_eq = f"$u(x,t) = {amplitude:.1f} \\sin(2\\pi \\cdot {frequency:.1f} \\cdot x/{wave_speed:.1f} - 2\\pi \\cdot {frequency:.1f} \\cdot t)$"
wave_eq_general = "$\\frac{\\partial^2 u}{\\partial t^2} = c^2 \\frac{\\partial^2 u}{\\partial x^2}$"

# Set titles with equations
ax1.set_title(f'Wave Function Solution\n{wave_eq}', fontsize=11)
#ax2.set_title(f'Spatial Second Derivative (∂²u/∂x²)\n$\\nabla^2 u = \\frac{\\partial^2 u}{\\partial x^2}$', fontsize=11)

ax1.set_xlabel('Position (x)')
ax1.set_ylabel('Time (t)')
ax1.set_zlabel('Displacement u(x,t)')

ax2.set_xlabel('Position (x)')
ax2.set_ylabel('Time (t)')
ax2.set_zlabel('∂²u/∂x²')

# Add sliders
plt.subplots_adjust(bottom=0.3)

ax_speed = plt.axes([0.25, 0.18, 0.5, 0.03])
ax_freq = plt.axes([0.25, 0.12, 0.5, 0.03])
ax_amp = plt.axes([0.25, 0.06, 0.5, 0.03])

slider_speed = Slider(ax_speed, 'Wave Speed (c)', 0.5, 3.0, valinit=wave_speed)
slider_freq = Slider(ax_freq, 'Frequency (f)', 0.1, 2.0, valinit=frequency)
slider_amp = Slider(ax_amp, 'Amplitude (A)', 0.1, 2.0, valinit=amplitude)

# Add text area for wave equation explanation
fig.text(0.5, 0.25, 
         "Wave Equation: " + wave_eq_general + "\n" + 
         "Traveling wave solution: $u(x,t) = A \\sin(kx - \\omega t)$ where $k = 2\\pi f/c$ and $\\omega = 2\\pi f$\n" +
         "The Laplacian (in 1D): $\\nabla^2 u = \\frac{\\partial^2 u}{\\partial x^2} = -k^2 u$",
         ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

def update(val):
    # Get current slider values
    c = slider_speed.val
    f = slider_freq.val
    A = slider_amp.val
    
    # Update the surfaces
    ax1.clear()
    ax2.clear()
    
    Z_wave = wave_function(X, T, c, f, A)
    Z_laplacian = laplacian_wave(X, T, c, f, A)
    
    ax1.plot_surface(X, T, Z_wave, cmap='viridis', alpha=0.8)
    ax2.plot_surface(X, T, Z_laplacian, cmap='coolwarm', alpha=0.8)
    
    # Update equation text
    wave_eq = f"$u(x,t) = {A:.1f} \\sin(2\\pi \\cdot {f:.1f} \\cdot x/{c:.1f} - 2\\pi \\cdot {f:.1f} \\cdot t)$"
    
    # Update titles
    ax1.set_title(f'Wave Function Solution\n{wave_eq}', fontsize=11)
    #ax2.set_title(f'Spatial Second Derivative (∂²u/∂x²)\n$\\nabla^2 u = \\frac{\\partial^2 u}{\\partial x^2}$', fontsize=11)
    
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Time (t)')
    ax1.set_zlabel('Displacement u(x,t)')
    
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Time (t)')
    ax2.set_zlabel('∂²u/∂x²')
    
    # Set consistent z-limits
    max_z_wave = np.max(np.abs(Z_wave))
    ax1.set_zlim(-max_z_wave * 1.1, max_z_wave * 1.1)
    
    max_z_laplacian = np.max(np.abs(Z_laplacian))
    ax2.set_zlim(-max_z_laplacian * 1.1, max_z_laplacian * 1.1)
    
    fig.canvas.draw_idle()

slider_speed.on_changed(update)
slider_freq.on_changed(update)
slider_amp.on_changed(update)

plt.tight_layout(rect=[0, 0.3, 1, 0.95])
plt.show()