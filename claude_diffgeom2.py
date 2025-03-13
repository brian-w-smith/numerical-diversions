import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Set up the figure with a high-resolution and good size
plt.figure(figsize=(14, 12), dpi=120)
ax = plt.axes(projection='3d')

# Create parameter space
u = np.linspace(-3, 3, 150)
v = np.linspace(-3, 3, 150)
u, v = np.meshgrid(u, v)

# Create a complex surface combining multiple geometric elements
def complex_surface(u, v):
    # Base shape: modified Enneper surface with additional terms
    r = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v, u)
    
    # Damping factor to prevent extreme values
    damp = np.exp(-0.1 * r**2)
    
    # Base Enneper-like terms
    x = u - u**3/3 + u*v**2
    y = v - v**3/3 + v*u**2
    z = u**2 - v**2
    
    # Add wave patterns with frequency modulation
    wave_amp = 0.3
    x += wave_amp * np.sin(2*u + np.sin(v)) * damp
    y += wave_amp * np.sin(2*v + np.sin(u)) * damp
    
    # Add helicoid-inspired twist
    twist = 0.1 * r * np.sin(3*theta) * damp
    z += twist
    
    # Add radial oscillations
    radial_term = 0.2 * np.sin(4*r) * damp
    x += radial_term * np.cos(theta)
    y += radial_term * np.sin(theta)
    
    # Add vertical modulation based on hyperbolic functions
    z += 0.5 * np.tanh(r - 2) * np.cos(4*theta) * damp
    
    return x, y, z

# Function to compute an approximation of the Gaussian curvature
def approximate_gaussian_curvature(x, y, z, h=0.1):
    # Approximate the Gaussian curvature using finite differences
    # This is a simplified approach that works for visualization purposes
    n, m = x.shape
    K = np.zeros((n, m))
    
    # Interior points only
    for i in range(1, n-1):
        for j in range(1, m-1):
            # First derivatives
            x_u = (x[i+1, j] - x[i-1, j]) / (2*h)
            x_v = (x[i, j+1] - x[i, j-1]) / (2*h)
            y_u = (y[i+1, j] - y[i-1, j]) / (2*h)
            y_v = (y[i, j+1] - y[i, j-1]) / (2*h)
            z_u = (z[i+1, j] - z[i-1, j]) / (2*h)
            z_v = (z[i, j+1] - z[i, j-1]) / (2*h)
            
            # Second derivatives
            x_uu = (x[i+1, j] - 2*x[i, j] + x[i-1, j]) / (h**2)
            x_vv = (x[i, j+1] - 2*x[i, j] + x[i, j-1]) / (h**2)
            x_uv = (x[i+1, j+1] - x[i+1, j-1] - x[i-1, j+1] + x[i-1, j-1]) / (4*h**2)
            
            y_uu = (y[i+1, j] - 2*y[i, j] + y[i-1, j]) / (h**2)
            y_vv = (y[i, j+1] - 2*y[i, j] + y[i, j-1]) / (h**2)
            y_uv = (y[i+1, j+1] - y[i+1, j-1] - y[i-1, j+1] + y[i-1, j-1]) / (4*h**2)
            
            z_uu = (z[i+1, j] - 2*z[i, j] + z[i-1, j]) / (h**2)
            z_vv = (z[i, j+1] - 2*z[i, j] + z[i, j-1]) / (h**2)
            z_uv = (z[i+1, j+1] - z[i+1, j-1] - z[i-1, j+1] + z[i-1, j-1]) / (4*h**2)
            
            # Normal vector components using cross product of tangent vectors
            E = x_u**2 + y_u**2 + z_u**2
            F = x_u*x_v + y_u*y_v + z_u*z_v
            G = x_v**2 + y_v**2 + z_v**2
            
            L = x_uu*z_u*y_v + y_uu*z_v*x_u + z_uu*x_v*y_u - x_uu*y_u*z_v - y_uu*x_v*z_u - z_uu*y_v*x_u
            M = x_uv*z_u*y_v + y_uv*z_v*x_u + z_uv*x_v*y_u - x_uv*y_u*z_v - y_uv*x_v*z_u - z_uv*y_v*x_u
            N = x_vv*z_u*y_v + y_vv*z_v*x_u + z_vv*x_v*y_u - x_vv*y_u*z_v - y_vv*x_v*z_u - z_vv*y_v*x_u
            
            # Gaussian curvature formula: (LN-M²)/(EG-F²)
            denominator = E*G - F**2
            if abs(denominator) > 1e-10:  # Avoid division by zero
                K[i, j] = (L*N - M**2) / denominator
            else:
                K[i, j] = 0
    
    # Apply some smoothing to the curvature field
    from scipy.ndimage import gaussian_filter
    K = gaussian_filter(K, sigma=1.0)
    
    return K

# Create our complex surface
X, Y, Z = complex_surface(u, v)

# Calculate Gaussian curvature (approximate)
K = approximate_gaussian_curvature(X, Y, Z)

# Clip extreme values for better visualization
k_min, k_max = np.percentile(K[~np.isnan(K)], [2, 98])
K = np.clip(K, k_min, k_max)

# Create a custom colormap that transitions through multiple colors
colors = plt.cm.plasma(np.linspace(0, 1, 256))
custom_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('custom_plasma', colors)

# Create a normalization for the color mapping
norm = plt.Normalize(K.min(), K.max())

# Plot the surface with colors based on Gaussian curvature
surf = ax.plot_surface(X, Y, Z, facecolors=custom_cmap(norm(K)),
                      alpha=0.9, linewidth=0, antialiased=True, shade=True)

# Add a colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap),
                   ax=ax, shrink=0.6, pad=0.05)
cbar.set_label('Gaussian Curvature', fontsize=14, fontweight='bold')

# Add principal curvature direction streamlines (simplified approach for visualization)
# We'll use contour lines for u and v directions
skip = 10
for i in range(0, len(u), skip):
    ax.plot(X[i, :], Y[i, :], Z[i, :], color='white', linewidth=0.5, alpha=0.3)
    ax.plot(X[:, i], Y[:, i], Z[:, i], color='white', linewidth=0.5, alpha=0.3)

# Add some special geodesic curves (simplified approximation)
t = np.linspace(0, 2*np.pi, 200)
for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
    r = np.linspace(0, 2.5, 200)
    u_curve = r * np.cos(t + phase)
    v_curve = r * np.sin(t + phase)
    x_curve, y_curve, z_curve = complex_surface(u_curve, v_curve)
    ax.plot(x_curve, y_curve, z_curve, color='yellow', linewidth=1.5, alpha=0.7)

# Set viewing angle and labels
ax.view_init(elev=35, azim=60)
ax.set_xlabel('X', fontsize=14, fontweight='bold')
ax.set_ylabel('Y', fontsize=14, fontweight='bold')
ax.set_zlabel('Z', fontsize=14, fontweight='bold')
ax.set_title('Complex Differential Geometry Surface', fontsize=18, fontweight='bold')

# Make the background black for better contrast
ax.set_facecolor('black')
plt.gcf().set_facecolor('black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.title.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

# Create a second visualization: an animated view
from matplotlib.animation import FuncAnimation

fig2 = plt.figure(figsize=(14, 12), dpi=100)
ax2 = plt.axes(projection='3d')

# Use the same surface but add a time parameter for morphing
def morphing_complex_surface(u, v, t):
    X, Y, Z = complex_surface(u, v)
    
    # Add time-dependent morphing
    morph_factor = np.sin(t) * 0.4
    X += morph_factor * np.sin(2*u + t)
    Y += morph_factor * np.sin(2*v + t)
    Z += morph_factor * np.cos(u + v + t)
    
    return X, Y, Z

# Initial plot
X0, Y0, Z0 = morphing_complex_surface(u, v, 0)
K0 = approximate_gaussian_curvature(X0, Y0, Z0)
K0 = np.clip(K0, k_min, k_max)

norm2 = plt.Normalize(k_min, k_max)
surf2 = ax2.plot_surface(X0, Y0, Z0, facecolors=custom_cmap(norm2(K0)),
                        alpha=0.9, linewidth=0, antialiased=True, shade=True)

# Set viewing angle and labels for the second plot
ax2.view_init(elev=35, azim=60)
ax2.set_xlabel('X', fontsize=14, fontweight='bold')
ax2.set_ylabel('Y', fontsize=14, fontweight='bold')
ax2.set_zlabel('Z', fontsize=14, fontweight='bold')
ax2.set_title('Morphing Differential Geometry Surface', fontsize=18, fontweight='bold')

# Apply the same styling as the first plot
ax2.set_facecolor('black')
plt.gcf().set_facecolor('black')
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.grid(False)
ax2.xaxis.label.set_color('white')
ax2.yaxis.label.set_color('white')
ax2.zaxis.label.set_color('white')
ax2.title.set_color('white')
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')
ax2.tick_params(axis='z', colors='white')

# Add colorbar
cbar2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm2, cmap=custom_cmap),
                    ax=ax2, shrink=0.6, pad=0.05)
cbar2.set_label('Gaussian Curvature', fontsize=14, fontweight='bold')

# Set axis limits
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_zlim(-3, 3)

plt.tight_layout()
plt.show()

# To create an animation (uncomment to run):
'''
def update_surface(frame):
    ax2.clear()
    
    t = frame * 0.1
    X, Y, Z = morphing_complex_surface(u, v, t)
    K = approximate_gaussian_curvature(X, Y, Z)
    K = np.clip(K, k_min, k_max)
    
    surf = ax2.plot_surface(X, Y, Z, facecolors=custom_cmap(norm2(K)),
                           alpha=0.9, linewidth=0, antialiased=True, shade=True)
    
    # Set viewing angle and labels
    ax2.view_init(elev=35, azim=60 + frame)
    ax2.set_xlabel('X', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax2.set_zlabel('Z', fontsize=14, fontweight='bold')
    ax2.set_title('Morphing Differential Geometry Surface', fontsize=18, fontweight='bold')
    
    # Apply styling
    ax2.set_facecolor('black')
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.grid(False)
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.zaxis.label.set_color('white')
    ax2.title.set_color('white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='z', colors='white')
    
    # Set axis limits
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_zlim(-3, 3)
    
    return [surf]

ani = FuncAnimation(fig2, update_surface, frames=60, interval=100, blit=False)
ani.save('complex_surface_animation.gif', writer='pillow', fps=15, dpi=80)
'''