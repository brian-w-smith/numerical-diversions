import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

# Create a new figure with a 3D subplot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Set the background color
fig.patch.set_facecolor('#16213e')
ax.set_facecolor('#1a1a2e')

# Create the saddle surface (hyperbolic paraboloid)
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = X**2 - Y**2  # z = x² - y² equation for a hyperbolic paraboloid

# Create a custom colormap for the surface
colors = plt.cm.viridis(np.linspace(0, 1, 256))
custom_cmap = cm.colors.ListedColormap(colors)

# Apply a light source for 3D effect
ls = LightSource(azdeg=315, altdeg=45)
illuminated_surface = ls.shade(Z, cmap=custom_cmap, vert_exag=0.5, blend_mode='soft')

# Plot the surface with custom coloring
surf = ax.plot_surface(X, Y, Z, facecolors=illuminated_surface, 
                       alpha=0.8, linewidth=0, antialiased=True)

# Add parametric curves in the u-direction (red curves)
for j in [-4, -2, 0, 2, 4]:
    u = np.linspace(-5, 5, 100)
    x_u = u
    y_u = j * np.ones_like(u)
    z_u = x_u**2 - y_u**2
    ax.plot(x_u, y_u, z_u, color='#ff3366', linewidth=2)

# Add parametric curves in the v-direction (blue curves)
for i in [-4, -2, 0, 2, 4]:
    v = np.linspace(-5, 5, 100)
    x_v = i * np.ones_like(v)
    y_v = v
    z_v = x_v**2 - y_v**2
    ax.plot(x_v, y_v, z_v, color='#66ccff', linewidth=2)

# Mark a specific point on the surface (for tangent plane)
point_x, point_y = 2, 1
point_z = point_x**2 - point_y**2
ax.scatter([point_x], [point_y], [point_z], color='#ffcc00', s=100)

# Create a tangent plane at the selected point
# The tangent plane equation: z = z_0 + (∂z/∂x)(x-x_0) + (∂z/∂y)(y-y_0)
# For z = x² - y², ∂z/∂x = 2x and ∂z/∂y = -2y
dz_dx = 2 * point_x
dz_dy = -2 * point_y

# Generate tangent plane points
plane_size = 2
plane_x = np.linspace(point_x - plane_size, point_x + plane_size, 10)
plane_y = np.linspace(point_y - plane_size, point_y + plane_size, 10)
plane_X, plane_Y = np.meshgrid(plane_x, plane_y)
plane_Z = point_z + dz_dx * (plane_X - point_x) + dz_dy * (plane_Y - point_y)

# Plot the tangent plane
ax.plot_surface(plane_X, plane_Y, plane_Z, color='#44bb77', alpha=0.4, linewidth=0)

# Plot normal vector
normal_x = -dz_dx
normal_y = -dz_dy
normal_z = 1
normal_length = 2 / np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
ax.quiver(point_x, point_y, point_z,
          normal_x * normal_length, normal_y * normal_length, normal_z * normal_length,
          color='#ffcc00', arrow_length_ratio=0.1, linewidth=2)

# Plot tangent vectors
tangent_u_x, tangent_u_y, tangent_u_z = 1, 0, dz_dx
tangent_v_x, tangent_v_y, tangent_v_z = 0, 1, dz_dy

ax.quiver(point_x, point_y, point_z,
          tangent_u_x, tangent_u_y, tangent_u_z,
          color='#ff3366', arrow_length_ratio=0.1, linewidth=2)
ax.quiver(point_x, point_y, point_z,
          tangent_v_x, tangent_v_y, tangent_v_z,
          color='#66ccff', arrow_length_ratio=0.1, linewidth=2)

# Plot a geodesic (approximation)
t = np.linspace(0, 6, 100)
geo_x = 0.5 * t + 1
geo_y = 0.8 * np.sin(t) + 1
geo_z = geo_x**2 - geo_y**2
ax.plot(geo_x, geo_y, geo_z, color='#ff9900', linewidth=4, linestyle='dashed')

# Principal curvature directions at the point
# For a saddle surface z = x² - y², the principal curvatures are along x+y=0 and x-y=0
t_p = np.linspace(-1, 1, 20)
# Principal direction 1
pc1_x = point_x + t_p
pc1_y = point_y + t_p
pc1_z = point_z + dz_dx * (pc1_x - point_x) + dz_dy * (pc1_y - point_y)
ax.plot(pc1_x, pc1_y, pc1_z, color='#cc44ff', linewidth=3)

# Principal direction 2
pc2_x = point_x + t_p
pc2_y = point_y - t_p
pc2_z = point_z + dz_dx * (pc2_x - point_x) + dz_dy * (pc2_y - point_y)
ax.plot(pc2_x, pc2_y, pc2_z, color='#00ffaa', linewidth=3)

# Add text annotations for Gaussian curvature
ax.text(4, 3, 16, "K < 0", color='white', fontsize=14)

# Add labels
ax.text(point_x + 1, point_y, point_z + 2, "Tangent Plane", color='white', fontsize=12)
ax.text(geo_x[-1], geo_y[-1], geo_z[-1] + 2, "Geodesic", color='white', fontsize=12)
ax.text(-4, -4, 32, "Saddle Surface", color='white', fontsize=12)
ax.text(point_x + normal_x * normal_length, 
        point_y + normal_y * normal_length, 
        point_z + normal_z * normal_length + 1, "Normal", color='white', fontsize=12)
ax.text(pc1_x[-1], pc1_y[-1], pc1_z[-1] + 1, "Principal Curvature 1", color='#cc44ff', fontsize=10)
ax.text(pc2_x[-1], pc2_y[-1], pc2_z[-1] + 1, "Principal Curvature 2", color='#00ffaa', fontsize=10)

# Set view angle and limits
ax.view_init(elev=30, azim=-45)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-10, 30)

# Remove axis labels and ticks for cleaner visualization
ax.set_axis_off()

# Add a title
plt.title("Differential Geometry Visualization", color='white', fontsize=16, pad=20)

# Adjust figure margins
plt.tight_layout()

plt.show()