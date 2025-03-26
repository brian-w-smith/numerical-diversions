import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class FullMatrix3DTransformationVisualizer:
    def __init__(self):
        # Initial triangle coordinates (homogeneous coordinates)
        self.original_triangle = np.array([
            [0, 1, 0.5, 0],   # x coordinates
            [0, 0, 1, 0],     # y coordinates
            [1, 1, 1, 1]      # homogeneous coordinate
        ])
        
        # Create the figure with more vertical space
        self.fig = plt.figure(figsize=(18, 12))
        
        # Plots
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        # Adjust subplot layout
        plt.subplots_adjust(bottom=0.4, wspace=0.3, left=0.1, right=0.9)
        
        # Plot original triangle
        x_orig = np.append(self.original_triangle[0][:3], self.original_triangle[0][0])
        y_orig = np.append(self.original_triangle[1][:3], self.original_triangle[1][0])
        self.original_line, = self.ax1.plot(x_orig, y_orig, 'b-')
        self.ax1.set_title('Original Shape')
        self.ax1.set_xlim(-2, 2)
        self.ax1.set_ylim(-2, 2)
        self.ax1.grid(True)
        
        # Plot transformed triangle
        x_trans = np.append(self.original_triangle[0][:3], self.original_triangle[0][0])
        y_trans = np.append(self.original_triangle[1][:3], self.original_triangle[1][0])
        self.transformed_line, = self.ax2.plot(x_trans, y_trans, 'r-')
        self.ax2.set_title('Transformed Shape')
        self.ax2.set_xlim(-2, 2)
        self.ax2.set_ylim(-2, 2)
        self.ax2.grid(True)
        
        # Matrix display text
        self.matrix_text = self.fig.text(
            0.5, 0.02, 
            'Transformation Matrix:\n[ 1.0  0.0  0.0 ]\n[ 0.0  1.0  0.0 ]\n[ 0.0  0.0  1.0 ]', 
            horizontalalignment='center',
            verticalalignment='bottom',
            fontfamily='monospace',
            fontsize=9,
            bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7)
        )
        
        # Initial transformation matrix (identity matrix)
        self.matrix = np.eye(3)
        
        # Create sliders for all 9 matrix elements
        slider_color = 'lightgoldenrodyellow'
        
        # Slider configuration: (left, bottom, label, initial value)
        slider_params = [
            # First row
            (0.1, 0.35, 'M[0,0]', 1.0),
            (0.1, 0.32, 'M[0,1]', 0.0),
            (0.1, 0.29, 'M[0,2]', 0.0),
            
            # Second row
            (0.1, 0.24, 'M[1,0]', 0.0),
            (0.1, 0.21, 'M[1,1]', 1.0),
            (0.1, 0.18, 'M[1,2]', 0.0),
            
            # Third row
            (0.1, 0.13, 'M[2,0]', 0.0),
            (0.1, 0.10, 'M[2,1]', 0.0),
            (0.1, 0.07, 'M[2,2]', 1.0)
        ]
        
        self.sliders = []
        
        # Create sliders with improved spacing
        for left, bottom, label, initial_val in slider_params:
            ax_slider = plt.axes([left, bottom, 0.7, 0.02], facecolor=slider_color)
            slider = Slider(
                ax_slider, label, -2.0, 2.0, 
                valinit=initial_val, 
                valstep=0.1
            )
            slider.on_changed(self.update)
            self.sliders.append(slider)
        
    def update(self, val):
        # Update transformation matrix based on slider values
        self.matrix = np.array([
            [self.sliders[0].val, self.sliders[1].val, self.sliders[2].val],
            [self.sliders[3].val, self.sliders[4].val, self.sliders[5].val],
            [self.sliders[6].val, self.sliders[7].val, self.sliders[8].val]
        ])
        
        # Apply transformation
        transformed_triangle = self.matrix @ self.original_triangle
        
        # Normalize homogeneous coordinates
        transformed_triangle[:2] /= transformed_triangle[2]
        
        # Update transformed plot with closed triangle
        x_trans = np.append(transformed_triangle[0][:3], transformed_triangle[0][0])
        y_trans = np.append(transformed_triangle[1][:3], transformed_triangle[1][0])
        self.transformed_line.set_xdata(x_trans)
        self.transformed_line.set_ydata(y_trans)
        
        # Update matrix display text
        matrix_str = 'Transformation Matrix:\n'
        matrix_str += f'[ {self.matrix[0,0]:.1f}  {self.matrix[0,1]:.1f}  {self.matrix[0,2]:.1f} ]\n'
        matrix_str += f'[ {self.matrix[1,0]:.1f}  {self.matrix[1,1]:.1f}  {self.matrix[1,2]:.1f} ]\n'
        matrix_str += f'[ {self.matrix[2,0]:.1f}  {self.matrix[2,1]:.1f}  {self.matrix[2,2]:.1f} ]'
        self.matrix_text.set_text(matrix_str)
        
        # Adjust plot limits dynamically
        self.ax2.set_xlim(
            min(x_trans) - 1, 
            max(x_trans) + 1
        )
        self.ax2.set_ylim(
            min(y_trans) - 1, 
            max(y_trans) + 1
        )
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
    
    def show(self):
        plt.show()

# Create and display the visualization
if __name__ == '__main__':
    visualizer = FullMatrix3DTransformationVisualizer()
    visualizer.show()