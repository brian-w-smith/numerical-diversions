import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation

class NavierStokesLaplacianViz:
    def __init__(self):
        # Simulation parameters
        self.grid_size = 50
        self.viscosity = 0.01
        self.dt = 0.1
        self.iterations = 10
        
        # Initial velocity fields
        self.u = np.zeros((self.grid_size, self.grid_size))  # x-velocity
        self.v = np.zeros((self.grid_size, self.grid_size))  # y-velocity
        
        # Create initial vortex
        self.create_vortex()
        
        # Setup the visualization
        self.setup_plot()
        
    def create_vortex(self):
        """Create a vortex in the center of the domain"""
        center_x, center_y = self.grid_size // 2, self.grid_size // 2
        radius = self.grid_size // 6
        
        x = np.arange(0, self.grid_size)
        y = np.arange(0, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Distance from center
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Create vortex velocity field
        strength = 1.0
        self.u = -strength * (Y - center_y) * np.exp(-dist**2 / (2 * radius**2))
        self.v = strength * (X - center_x) * np.exp(-dist**2 / (2 * radius**2))
    
    def laplacian(self, field):
        """Compute the Laplacian of a field using finite differences"""
        lapl = np.zeros_like(field)
        
        # Interior points (no boundaries)
        lapl[1:-1, 1:-1] = (
            field[0:-2, 1:-1] +  # left
            field[2:, 1:-1] +    # right
            field[1:-1, 0:-2] +  # bottom
            field[1:-1, 2:] -    # top
            4 * field[1:-1, 1:-1]  # center
        )
        
        return lapl
    
    def diffuse(self):
        """Apply viscous diffusion using the Laplacian"""
        # Compute Laplacians
        lapl_u = self.laplacian(self.u)
        lapl_v = self.laplacian(self.v)
        
        # Store for visualization
        self.lapl_u = lapl_u
        self.lapl_v = lapl_v
        
        # Apply diffusion
        self.u += self.viscosity * lapl_u * self.dt
        self.v += self.viscosity * lapl_v * self.dt
    
    def setup_plot(self):
        """Set up the interactive plot"""
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.25)
        
        # Velocity plot
        self.ax_vel = self.axes[0]
        self.ax_vel.set_title('Velocity Field')
        self.ax_vel.set_xlabel('X')
        self.ax_vel.set_ylabel('Y')
        
        # Laplacian plot
        self.ax_lapl = self.axes[1]
        self.ax_lapl.set_title('Laplacian Magnitude')
        self.ax_lapl.set_xlabel('X')
        self.ax_lapl.set_ylabel('Y')
        
        # Setup plot elements
        x = np.arange(0, self.grid_size)
        y = np.arange(0, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Velocity visualization
        skip = 3
        self.quiver = self.ax_vel.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                                         self.u[::skip, ::skip], self.v[::skip, ::skip],
                                         scale=30)
        
        # Laplacian visualization
        lapl_mag = np.sqrt(self.laplacian(self.u)**2 + self.laplacian(self.v)**2)
        self.lapl_plot = self.ax_lapl.imshow(lapl_mag, cmap='viridis', 
                                            origin='lower', vmin=0, vmax=0.1)
        self.cbar = self.fig.colorbar(self.lapl_plot, ax=self.ax_lapl)
        
        # Add sliders for interaction
        ax_viscosity = plt.axes([0.25, 0.12, 0.65, 0.03])
        self.viscosity_slider = Slider(
            ax=ax_viscosity,
            label='Viscosity',
            valmin=0.001,
            valmax=0.1,
            valinit=self.viscosity,
            valstep=0.001
        )
        
        # Reset button
        ax_reset = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.reset_button = Button(ax_reset, 'Reset', color='lightgoldenrodyellow')
        
        # Play/Pause button
        ax_playpause = plt.axes([0.65, 0.02, 0.1, 0.04])
        self.playpause_button = Button(ax_playpause, 'Pause', color='lightgoldenrodyellow')
        
        # Animation controls
        self.paused = False
        
        # Connect events
        self.viscosity_slider.on_changed(self.update_viscosity)
        self.reset_button.on_clicked(self.reset)
        self.playpause_button.on_clicked(self.toggle_animation)
        
        # Animation
        self.anim = FuncAnimation(self.fig, self.update, interval=50, blit=False)
    
    def update_viscosity(self, val):
        """Update viscosity value from slider"""
        self.viscosity = val
    
    def reset(self, event):
        """Reset the simulation"""
        self.create_vortex()
        
    def toggle_animation(self, event):
        """Toggle animation on/off"""
        self.paused = not self.paused
        if self.paused:
            self.playpause_button.label.set_text('Play')
        else:
            self.playpause_button.label.set_text('Pause')
        self.playpause_button.ax.figure.canvas.draw_idle()
    
    def update(self, frame):
        """Update function for the animation"""
        if not self.paused:
            # Run multiple iterations per frame for smoother animation
            for _ in range(self.iterations):
                self.diffuse()
            
            # Update velocity quiver plot
            skip = 3
            x = np.arange(0, self.grid_size)
            y = np.arange(0, self.grid_size)
            X, Y = np.meshgrid(x, y)
            self.quiver.set_UVC(self.u[::skip, ::skip], self.v[::skip, ::skip])
            
            # Update Laplacian plot
            lapl_mag = np.sqrt(self.lapl_u**2 + self.lapl_v**2)
            self.lapl_plot.set_data(lapl_mag)
            
        return self.quiver, self.lapl_plot
    
    def show(self):
        """Show the plot"""
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    viz = NavierStokesLaplacianViz()
    viz.show()