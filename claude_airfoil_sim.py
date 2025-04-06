import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev

class AirfoilSimulation:
    def __init__(self, airfoil_type="NACA0012"):
        self.airfoil_type = airfoil_type
        self.angle_of_attack = 5  # degrees
        self.freestream_velocity = 10.0  # m/s
        self.grid_size = 100
        self.field_size = 4.0  # grid extends from -field_size to +field_size
        # Initialize simulation
        self.initialize_simulation()
        
        # Initialize plot attributes
        self.fig = None
        self.ax = None

    def generate_airfoil_shape(self):
        """Generate the airfoil shape based on the specified type."""
        if self.airfoil_type.startswith("NACA"):
            # Create a NACA 4-digit airfoil
            # For simplicity, we'll use NACA0012 with 12% thickness
            thickness = 0.12
            x = np.linspace(0, 1, 100)
            y_t = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 
                                  0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
            
            # Make symmetric airfoil
            x_upper = x
            y_upper = y_t
            x_lower = np.flip(x)
            y_lower = -np.flip(y_t)
            
            # Combine to form the complete airfoil
            x = np.concatenate((x_upper, x_lower[1:]))
            y = np.concatenate((y_upper, y_lower[1:]))
            
            # Rotate the airfoil by the angle of attack
            angle_rad = np.radians(self.angle_of_attack)
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            
            coordinates = np.vstack((x - 0.5, y)).T
            rotated = np.dot(coordinates, rotation_matrix.T)
            
            self.airfoil_x = rotated[:, 0]
            self.airfoil_y = rotated[:, 1]
        else:
            # Default simple elliptical airfoil if type not recognized
            t = np.linspace(0, 2*np.pi, 100)
            self.airfoil_x = 0.5 * np.cos(t)
            self.airfoil_y = 0.1 * np.sin(t)
    
    def initialize_simulation(self):
        """Set up simulation grid and initial flow field."""
        self.generate_airfoil_shape()
        
        # Create grid for velocity field
        x = np.linspace(-self.field_size, self.field_size, self.grid_size)
        y = np.linspace(-self.field_size, self.field_size, self.grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initialize velocity and pressure fields
        self.u = np.ones_like(self.X) * self.freestream_velocity * np.cos(np.radians(self.angle_of_attack))
        self.v = np.ones_like(self.Y) * self.freestream_velocity * np.sin(np.radians(self.angle_of_attack))
        self.pressure = np.zeros_like(self.X)
        
        # Create a mask for points inside the airfoil
        self.mask = self.create_airfoil_mask()
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
        
        # Calculate initial pressure field
        self.calculate_pressure()
    
    def create_airfoil_mask(self):
        """Create a mask identifying grid points inside the airfoil."""
        # Create a smoother representation of the airfoil boundary
        tck, u = splprep([self.airfoil_x, self.airfoil_y], s=0, per=True)
        smooth_x, smooth_y = splev(np.linspace(0, 1, 1000), tck)
        
        # Create a mask of points inside the airfoil
        mask = np.zeros_like(self.X, dtype=bool)
        
        # Simple point-in-polygon test for each grid point
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x, y = self.X[i, j], self.Y[i, j]
                if self.point_in_airfoil(x, y, smooth_x, smooth_y):
                    mask[i, j] = True
        
        return mask
    
    def point_in_airfoil(self, x, y, airfoil_x, airfoil_y):
        """Check if a point is inside the airfoil using ray casting algorithm."""
        n = len(airfoil_x)
        inside = False
        
        p1x, p1y = airfoil_x[0], airfoil_y[0]
        for i in range(1, n + 1):
            p2x, p2y = airfoil_x[i % n], airfoil_y[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def apply_boundary_conditions(self):
        """Apply boundary conditions for flow around the airfoil."""
        # Set velocity to zero inside the airfoil (no-slip condition)
        self.u[self.mask] = 0
        self.v[self.mask] = 0
        
        # Apply potential flow approximation around the airfoil
        # This is a simplified model that doesn't solve the full Navier-Stokes equations
        
        # Define source strength and location for flow around the airfoil
        source_strength = 5.0
        source_x, source_y = 0, 0
        
        # Add potential flow components
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if not self.mask[i, j]:
                    x, y = self.X[i, j], self.Y[i, j]
                    
                    # Distance from source
                    r = np.sqrt((x - source_x)**2 + (y - source_y)**2)
                    if r < 0.1:  # Avoid division by very small numbers
                        continue
                    
                    # Add freestream and source flow components
                    theta = np.arctan2(y - source_y, x - source_x)
                    
                    # Source contribution
                    u_source = source_strength * np.cos(theta) / (2 * np.pi * r)
                    v_source = source_strength * np.sin(theta) / (2 * np.pi * r)
                    
                    # Add doublet effect for cylinder-like flow
                    u_doublet = -source_strength * np.cos(theta) * 0.25**2 / (2 * np.pi * r**2)
                    v_doublet = -source_strength * np.sin(theta) * 0.25**2 / (2 * np.pi * r**2)
                    
                    # Update velocity field
                    self.u[i, j] += u_source + u_doublet
                    self.v[i, j] += v_source + v_doublet
    
    def calculate_pressure(self):
        """Calculate pressure field using Bernoulli's equation."""
        # Calculate velocity magnitude
        velocity_mag = np.sqrt(self.u**2 + self.v**2)
        
        # Calculate pressure using Bernoulli's equation (simplified)
        # P + 0.5*rho*V^2 = constant
        # We'll set the constant so that freestream pressure = 0
        rho = 1.0  # Air density
        self.pressure = 0.5 * rho * (self.freestream_velocity**2 - velocity_mag**2)
    
    def init_animation(self):
        """Initialize animation function."""
        # Create the initial plots - we'll update these in the animation
        self.pressure_contour = self.ax.contourf(self.X, self.Y, self.pressure, levels=20, cmap='RdBu_r')
        self.streamline = self.ax.streamplot(self.X, self.Y, self.u, self.v, density=1, color='black', linewidth=0.5)
        self.airfoil_line, = self.ax.plot(self.airfoil_x, self.airfoil_y, 'k-', linewidth=2)
        
        return self.airfoil_line,
    
    def update_animation(self, frame):
        """Update function for animation."""
        # Update angle of attack for this frame
        self.angle_of_attack = 5 + 3 * np.sin(frame / 10)
        
        # Reinitialize the simulation with the new angle
        self.initialize_simulation()
        
        # Clear the axis and redraw everything
        self.ax.clear()
        
        # Redraw all plots
        self.pressure_contour = self.ax.contourf(self.X, self.Y, self.pressure, levels=20, cmap='RdBu_r')
        self.streamline = self.ax.streamplot(self.X, self.Y, self.u, self.v, density=1, color='black', linewidth=0.5)
        self.airfoil_line, = self.ax.plot(self.airfoil_x, self.airfoil_y, 'k-', linewidth=2)
        
        # Update axis properties
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Airfoil Simulation: Angle of Attack = {self.angle_of_attack:.1f}°')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        return self.airfoil_line,
    
    def visualize(self, animate=True):
        """Visualize the flow field and pressure distribution."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Set axis properties
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Airfoil Simulation: {self.airfoil_type}, Angle of Attack = {self.angle_of_attack}°')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        if animate:
            # Create animation
            ani = FuncAnimation(
                self.fig, 
                self.update_animation, 
                frames=100, 
                interval=100, 
                blit=False,
                init_func=self.init_animation
            )
            plt.colorbar(self.ax.contourf(self.X, self.Y, self.pressure, levels=20, cmap='RdBu_r'),
                         ax=self.ax, label='Pressure')
            plt.tight_layout()
            plt.show()
            return ani
        else:
            # For static visualization
            pressure_contour = self.ax.contourf(self.X, self.Y, self.pressure, levels=20, cmap='RdBu_r')
            plt.colorbar(pressure_contour, ax=self.ax, label='Pressure')
            self.ax.streamplot(self.X, self.Y, self.u, self.v, density=1, color='black', linewidth=0.5)
            self.ax.plot(self.airfoil_x, self.airfoil_y, 'k-', linewidth=2)
            plt.tight_layout()
            plt.show()

# Run the simulation
if __name__ == "__main__":
    # Create and run simulation with NACA0012 airfoil
    sim = AirfoilSimulation(airfoil_type="NACA0012")
    sim.visualize(animate=True)