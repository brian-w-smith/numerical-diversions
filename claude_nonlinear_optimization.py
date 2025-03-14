import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors

# Create figure
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Set the view limits
x_min, x_max = 0, 5
y_min, y_max = 0, 5
z_min, z_max = 0, 5

# Create meshgrid for plotting
resolution = 100  # Higher resolution for smoother surfaces
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

# Define the two nonlinear constraints we're keeping
# 1. Quadratic constraint: x²/4 + y²/9 + z ≤ 5
def constraint1(x, y):
    return 5 - x**2/4 - y**2/9

# 2. Exponential constraint: e^(x/5) + y + z ≤ 6
def constraint2(x, y):
    return 6 - np.exp(x/5) - y

# Calculate Z values for each constraint
Z1 = constraint1(X, Y)
Z2 = constraint2(X, Y)

# Ensure Z values are within bounds and replace NaN/Inf values
def clean_data(Z):
    Z = np.maximum(Z, z_min)
    Z = np.minimum(Z, z_max)
    Z[~np.isfinite(Z)] = z_min
    return Z

Z1 = clean_data(Z1)
Z2 = clean_data(Z2)

# Function to calculate feasible region (intersection of constraints)
def get_feasible_z(x, y):
    c1 = constraint1(x, y)
    c2 = constraint2(x, y)
    
    # The minimum of all constraints (upper bound of z)
    z_upper = np.minimum(c1, c2)
    return np.maximum(z_upper, z_min)

# Calculate Z for feasible region
Z_feasible = get_feasible_z(X, Y)
Z_feasible = clean_data(Z_feasible)

# Plot the constraint surfaces with increased transparency for better visibility
colors_list = ['red', 'blue']
alpha = 0.4

# Plot each constraint surface
surf1 = ax.plot_surface(X, Y, Z1, alpha=alpha, color=colors_list[0], label='Constraint 1: x²/4 + y²/9 + z ≤ 5')
surf2 = ax.plot_surface(X, Y, Z2, alpha=alpha, color=colors_list[1], label='Constraint 2: e^(x/5) + y + z ≤ 6')

# Add nonnegativity constraints (coordinate planes)
# xy-plane (z=0)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.15, color='gray', label='z ≥ 0')

# xz-plane (y=0)
xx, zz = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
yy = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.15, color='gray')

# yz-plane (x=0)
yy, zz = np.meshgrid(np.linspace(y_min, y_max, 10), np.linspace(z_min, z_max, 10))
xx = np.zeros_like(yy)
ax.plot_surface(xx, yy, zz, alpha=0.15, color='gray')

# Highlight the intersection curve of the two constraint surfaces
intersection_points = []
for i in range(resolution):
    for j in range(resolution):
        x_val = X[i, j]
        y_val = Y[i, j]
        z1_val = Z1[i, j]
        z2_val = Z2[i, j]
        # If the two constraint values are very close, it's an intersection point
        if abs(z1_val - z2_val) < 0.05 and z1_val > 0:
            intersection_points.append((x_val, y_val, z1_val))

# Convert intersection points to array for scatter plot
if intersection_points:
    intersect_points = np.array(intersection_points)
    ax.scatter(intersect_points[:, 0], intersect_points[:, 1], intersect_points[:, 2], 
               color='green', s=20, alpha=1, label='Constraint Intersection')

# Highlight the feasible region boundary
boundary_points = []
for i in range(resolution):
    for j in range(resolution):
        if Z_feasible[i, j] > 0:
            x_val = X[i, j]
            y_val = Y[i, j]
            z_val = Z_feasible[i, j]
            boundary_points.append((x_val, y_val, z_val))

# Convert boundary points to array for scatter plot
if boundary_points:
    bound_points = np.array(boundary_points)
    ax.scatter(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], 
               color='cyan', s=3, alpha=0.3)

# Objective function: maximize f(x,y,z) = 2x + 3y + z
# For visualization, let's represent it as planes at specific values
def objective_plane(value):
    # Solve 2x + 3y + z = value for z
    X_obj, Y_obj = np.meshgrid(np.linspace(x_min, x_max, 5), np.linspace(y_min, y_max, 5))
    Z_obj = value - 2*X_obj - 3*Y_obj
    return X_obj, Y_obj, Z_obj

# Plot objective function planes at different values
for val in [5, 10, 15]:
    X_obj, Y_obj, Z_obj = objective_plane(val)
    ax.plot_surface(X_obj, Y_obj, Z_obj, alpha=0.15, color='yellow')

# Find an approximate optimal solution by sampling the feasible region
optimal_point = None
max_obj_value = -float('inf')

sample_points = []
for i in range(resolution):
    for j in range(resolution):
        x_val = X[i, j]
        y_val = Y[i, j]
        # Loop through possible z values
        for z_ratio in np.linspace(0, 1, 10):
            z_val = z_ratio * Z_feasible[i, j]
            if z_val >= 0:  # Check z ≥ 0 constraint
                obj_value = 2*x_val + 3*y_val + z_val
                sample_points.append((x_val, y_val, z_val, obj_value))
                if obj_value > max_obj_value:
                    max_obj_value = obj_value
                    optimal_point = (x_val, y_val, z_val)

# Plot the optimal point
if optimal_point:
    ax.scatter(*optimal_point, color='red', s=200, marker='*', 
              label=f'Optimal Point ({optimal_point[0]:.2f}, {optimal_point[1]:.2f}, {optimal_point[2]:.2f})')
    
    # Add a marker for objective direction
    ax.quiver(*optimal_point, 2, 3, 1, color='black', arrow_length_ratio=0.1, 
             length=1, label='Objective Direction (2,3,1)')

# Set labels and title
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('z', fontsize=12)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
ax.set_title('3D Optimization with Two Nonlinear Constraints', fontsize=16)

# Add text annotations for constraint equations
ax.text(1, 1, 4.5, 'x²/4 + y²/9 + z ≤ 5', color='red', fontsize=12)
ax.text(3, 2, 2.5, 'e^(x/5) + y + z ≤ 6', color='blue', fontsize=12)
ax.text(3, 4, 3, 'Maximize: 2x + 3y + z', color='black', fontsize=12)

# Adjust the view angle for better visualization
ax.view_init(elev=25, azim=130)

# Add a custom legend
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color=colors_list[0], lw=4),
    Line2D([0], [0], color=colors_list[1], lw=4),
    Line2D([0], [0], color='green', marker='o', markersize=6, linestyle='None'),
    Line2D([0], [0], color='yellow', lw=4),
    Line2D([0], [0], color='red', marker='*', markersize=10, linestyle='None')
]

custom_labels = [
    'Constraint 1: x²/4 + y²/9 + z ≤ 5',
    'Constraint 2: e^(x/5) + y + z ≤ 6',
    'Constraint Intersection',
    'Objective Function Planes',
    'Optimal Solution'
]

ax.legend(custom_lines, custom_labels, loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()