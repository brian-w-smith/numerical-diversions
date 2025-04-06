import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

def plot_cross_product(v1, v2):
    # Calculate cross product
    cross = np.cross(v1, v2)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot origin
    ax.scatter([0], [0], [0], color='k', s=100)
    
    # Plot the two input vectors
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='Vector 1')
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='Vector 2')
    
    # Plot the cross product
    ax.quiver(0, 0, 0, cross[0], cross[1], cross[2], color='g', label='Cross Product')
    
    # Plot the plane spanned by v1 and v2
    # Create a meshgrid in the plane
    max_val = max(np.max(np.abs(v1)), np.max(np.abs(v2))) * 1.2
    xx, yy = np.meshgrid(np.linspace(-max_val, max_val, 10), 
                         np.linspace(-max_val, max_val, 10))
    
    # Find a normal vector to the plane (normalized cross product)
    if not np.allclose(cross, np.zeros(3)):  # Check if cross product is not zero
        normal = cross / np.linalg.norm(cross)
        
        # Equation of the plane: ax + by + cz + d = 0
        # where [a, b, c] is the normal vector and d is the distance from origin
        # Since the plane passes through origin, d = 0
        # Therefore z = -(ax + by) / c
        
        if abs(normal[2]) > 1e-6:  # Avoid division by zero
            z = -(normal[0] * xx + normal[1] * yy) / normal[2]
            # Only show part of the plane within reasonable bounds
            mask = (z > -max_val) & (z < max_val)
            xx_masked = np.where(mask, xx, np.nan)
            yy_masked = np.where(mask, yy, np.nan)
            z_masked = np.where(mask, z, np.nan)
            ax.plot_surface(xx_masked, yy_masked, z_masked, alpha=0.2, color='gray')
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    
    # Add legend
    ax.legend()
    
    # Add title with vector values and cross product result
    title = f"Vector 1: {v1}\nVector 2: {v2}\nCross Product: {cross}"
    plt.title(title)
    
    return fig, ax

def interactive_cross_product():
    # Initial vectors
    v1_init = np.array([1.0, 0.0, 0.0])
    v2_init = np.array([0.0, 1.0, 0.0])
    
    # Create figure
    fig, ax = plot_cross_product(v1_init, v2_init)
    
    # Add sliders for vector components
    plt.subplots_adjust(bottom=0.35)
    
    # Axes for sliders
    ax_v1x = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_v1y = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_v1z = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_v2x = plt.axes([0.25, 0.10, 0.65, 0.03])
    ax_v2y = plt.axes([0.25, 0.05, 0.65, 0.03])
    ax_v2z = plt.axes([0.25, 0.00, 0.65, 0.03])
    
    # Create sliders
    slider_v1x = Slider(ax_v1x, 'V1 X', -3.0, 3.0, valinit=v1_init[0])
    slider_v1y = Slider(ax_v1y, 'V1 Y', -3.0, 3.0, valinit=v1_init[1])
    slider_v1z = Slider(ax_v1z, 'V1 Z', -3.0, 3.0, valinit=v1_init[2])
    slider_v2x = Slider(ax_v2x, 'V2 X', -3.0, 3.0, valinit=v2_init[0])
    slider_v2y = Slider(ax_v2y, 'V2 Y', -3.0, 3.0, valinit=v2_init[1])
    slider_v2z = Slider(ax_v2z, 'V2 Z', -3.0, 3.0, valinit=v2_init[2])
    
    # Update function
    def update(val):
        # Clear current plot
        ax.clear()
        
        # Get current values from sliders
        v1 = np.array([slider_v1x.val, slider_v1y.val, slider_v1z.val])
        v2 = np.array([slider_v2x.val, slider_v2y.val, slider_v2z.val])
        
        # Calculate cross product
        cross = np.cross(v1, v2)
        
        # Plot vectors
        max_val = max(np.max(np.abs(v1)), np.max(np.abs(v2)), np.max(np.abs(cross))) * 1.2
        if max_val < 0.1:  # Ensure we have some minimum scale
            max_val = 3.0
            
        # Plot origin
        ax.scatter([0], [0], [0], color='k', s=100)
        
        # Plot the two input vectors
        ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='Vector 1')
        ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='Vector 2')
        
        # Plot the cross product
        ax.quiver(0, 0, 0, cross[0], cross[1], cross[2], color='g', label='Cross Product')
        
        # Plot the plane if vectors aren't parallel
        if not np.allclose(cross, np.zeros(3)):
            # Find a normal vector to the plane (normalized cross product)
            normal = cross / np.linalg.norm(cross)
            
            # Create a meshgrid in the plane
            xx, yy = np.meshgrid(np.linspace(-max_val, max_val, 10), 
                               np.linspace(-max_val, max_val, 10))
            
            if abs(normal[2]) > 1e-6:  # Avoid division by zero
                z = -(normal[0] * xx + normal[1] * yy) / normal[2]
                # Only show part of the plane within reasonable bounds
                mask = (z > -max_val) & (z < max_val)
                xx_masked = np.where(mask, xx, np.nan)
                yy_masked = np.where(mask, yy, np.nan)
                z_masked = np.where(mask, z, np.nan)
                ax.plot_surface(xx_masked, yy_masked, z_masked, alpha=0.2, color='gray')
            elif abs(normal[1]) > 1e-6:  # If normal[2] is zero, try using y
                z = np.zeros_like(xx)
                y = -(normal[0] * xx + normal[2] * z) / normal[1]
                mask = (y > -max_val) & (y < max_val)
                xx_masked = np.where(mask, xx, np.nan)
                y_masked = np.where(mask, y, np.nan)
                z_masked = np.where(mask, z, np.nan)
                ax.plot_surface(xx_masked, y_masked, z_masked, alpha=0.2, color='gray')
            elif abs(normal[0]) > 1e-6:  # If normal[1] is also zero, use x
                z = np.zeros_like(yy)
                x = -(normal[1] * yy + normal[2] * z) / normal[0]
                mask = (x > -max_val) & (x < max_val)
                x_masked = np.where(mask, x, np.nan)
                yy_masked = np.where(mask, yy, np.nan)
                z_masked = np.where(mask, z, np.nan)
                ax.plot_surface(x_masked, yy_masked, z_masked, alpha=0.2, color='gray')
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        
        # Add legend
        ax.legend()
        
        # Add title with vector values and cross product result
        mag_cross = np.linalg.norm(cross)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)
        
        # Calculate the angle between vectors in degrees
        if mag_v1 > 1e-6 and mag_v2 > 1e-6:
            dot_product = np.dot(v1, v2)
            cos_angle = dot_product / (mag_v1 * mag_v2)
            # Handle floating point errors
            cos_angle = max(min(cos_angle, 1.0), -1.0)
            angle = np.degrees(np.arccos(cos_angle))
        else:
            angle = 0
            
        title = (f"Vector 1: [{v1[0]:.1f}, {v1[1]:.1f}, {v1[2]:.1f}], |V1|={mag_v1:.2f}\n"
                f"Vector 2: [{v2[0]:.1f}, {v2[1]:.1f}, {v2[2]:.1f}], |V2|={mag_v2:.2f}\n"
                f"Cross Product: [{cross[0]:.1f}, {cross[1]:.1f}, {cross[2]:.1f}], |V1×V2|={mag_cross:.2f}\n"
                f"Angle between vectors: {angle:.1f}°")
        ax.set_title(title)
        
        # Redraw
        fig.canvas.draw_idle()
    
    # Register the update function with each slider
    slider_v1x.on_changed(update)
    slider_v1y.on_changed(update)
    slider_v1z.on_changed(update)
    slider_v2x.on_changed(update)
    slider_v2y.on_changed(update)
    slider_v2z.on_changed(update)
    
    plt.show()

# Run the interactive visualization
if __name__ == "__main__":
    interactive_cross_product()