from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
import math
import numpy as np

class EulersEquationVisualization(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Set up the camera
        self.disableMouse()
        self.camera.setPos(0, -15, 5)
        self.camera.lookAt(0, 0, 0)
        
        # Create ambient lighting
        alight = AmbientLight('alight')
        alight.setColor((0.3, 0.3, 0.3, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        # Create directional lighting
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.7, 0.7, 0.7, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)
        
        # Add title text
        self.title = OnscreenText(text="Euler's Equation: e^(iπ) + 1 = 0",
                                 style=1, fg=(1, 1, 1, 1), pos=(0, 0.9),
                                 scale=0.07)
        
        # Create the complex plane
        self.create_complex_plane()
        
        # Create the unit circle
        self.create_unit_circle()
        
        # Create the exponential curve
        self.create_exponential_curve()
        
        # Add key point markers
        self.add_key_points()
        
        # Add description text
        self.description = OnscreenText(
            text="This visualization shows how e^(ix) traces the unit circle in the complex plane.\n"
                 "When x = π, e^(iπ) = -1, thus e^(iπ) + 1 = 0",
            style=1, fg=(1, 1, 1, 1), pos=(0, -0.9), scale=0.05)
        
        # Add rotation task
        self.taskMgr.add(self.rotate_camera_task, "RotateCameraTask")
        self.angle = 0
        
    def create_complex_plane(self):
        # Create a grid for the complex plane
        grid_size = 5
        grid_spacing = 1.0
        
        # Create lines maker
        lines = LineSegs()
        lines.setThickness(1)
        
        # X-axis (real axis)
        lines.setColor(1, 0, 0, 1)  # Red for real axis
        lines.moveTo(-grid_size, 0, 0)
        lines.drawTo(grid_size, 0, 0)
        
        # Y-axis (imaginary axis)
        lines.setColor(0, 1, 0, 1)  # Green for imaginary axis
        lines.moveTo(0, 0, -grid_size)
        lines.drawTo(0, 0, grid_size)
        
        # Grid lines
        lines.setColor(0.5, 0.5, 0.5, 0.3)
        for i in range(-grid_size, grid_size + 1):
            if i != 0:  # Skip center lines (already drawn)
                # Parallel to x-axis
                lines.moveTo(-grid_size, 0, i * grid_spacing)
                lines.drawTo(grid_size, 0, i * grid_spacing)
                
                # Parallel to y-axis
                lines.moveTo(i * grid_spacing, 0, -grid_size)
                lines.drawTo(i * grid_spacing, 0, grid_size)
        
        # Create and attach the node
        node = lines.create()
        self.complex_plane = self.render.attachNewNode(node)
        
        # Add labels
        for i in range(-grid_size, grid_size + 1):
            if i != 0:
                # Real axis labels
                text = TextNode(f'label_real_{i}')
                text.setText(str(i))
                text.setTextColor(1, 0, 0, 1)
                text_np = self.render.attachNewNode(text)
                text_np.setPos(i, 0, -0.3)
                text_np.setScale(0.3)
                text_np.setBillboardPointEye()
                
                # Imaginary axis labels
                text = TextNode(f'label_imag_{i}')
                text.setText(f'{i}i')
                text.setTextColor(0, 1, 0, 1)
                text_np = self.render.attachNewNode(text)
                text_np.setPos(-0.3, 0, i)
                text_np.setScale(0.3)
                text_np.setBillboardPointEye()
        
        # Origin label
        text = TextNode('label_origin')
        text.setText('0')
        text.setTextColor(1, 1, 1, 1)
        text_np = self.render.attachNewNode(text)
        text_np.setPos(-0.3, 0, -0.3)
        text_np.setScale(0.3)
        text_np.setBillboardPointEye()
    
    def create_unit_circle(self):
        # Create a unit circle in the complex plane
        segments = 100
        lines = LineSegs()
        lines.setThickness(2)
        lines.setColor(0, 0, 1, 1)  # Blue for unit circle
        
        prev_x = 1
        prev_z = 0
        
        for i in range(1, segments + 1):
            angle = 2 * math.pi * i / segments
            x = math.cos(angle)
            z = math.sin(angle)
            
            if i == 1:
                lines.moveTo(prev_x, 0, prev_z)
            
            lines.drawTo(x, 0, z)
            prev_x, prev_z = x, z
        
        node = lines.create()
        self.unit_circle = self.render.attachNewNode(node)
        
        # Add unit circle label
        text = TextNode('unit_circle_label')
        text.setText('Unit Circle: |z| = 1')
        text.setTextColor(0, 0, 1, 1)
        text_np = self.render.attachNewNode(text)
        text_np.setPos(1.2, 0, 1.2)
        text_np.setScale(0.3)
        text_np.setBillboardPointEye()
    
    def create_exponential_curve(self):
        # Create the e^(ix) curve visualization with vector
        self.angle_marker = self.loader.loadModel("models/misc/sphere")  # Updated path
        self.angle_marker.setScale(0.05, 0.05, 0.05)
        self.angle_marker.setColor(1, 1, 0, 1)  # Yellow
        self.angle_marker.reparentTo(self.render)
        
        # Create a line to show the current vector
        self.vector_line = LineSegs()
        self.vector_line.setThickness(4)
        self.vector_line.setColor(1, 1, 0, 1)  # Yellow
        self.vector_line.moveTo(0, 0, 0)
        self.vector_line.drawTo(1, 0, 0)  # Start at 1 + 0i
        node = self.vector_line.create()
        self.vector = self.render.attachNewNode(node)
        
        # Add animated task for e^(ix) visualization
        self.phase = 0
        self.taskMgr.add(self.update_exponential, "UpdateExponential")
        
        # Create the exponential path
        path_segments = 100
        lines = LineSegs()
        lines.setThickness(3)
        lines.setColor(1, 0.5, 0, 1)  # Orange for the path
        
        lines.moveTo(1, 0, 0)  # Start at 1 + 0i (e^(i0))
        
        for i in range(1, path_segments + 1):
            angle = math.pi * i / path_segments
            x = math.cos(angle)
            z = math.sin(angle)
            lines.drawTo(x, 0, z)
        
        node = lines.create()
        self.exp_path = self.render.attachNewNode(node)
    
    def add_key_points(self):
        # Add markers for key points: 1, i, -1, -i
        key_points = [
            (1, 0, "e^(i·0) = 1", (1, 1, 1, 1)),
            (0, 1, "e^(i·π/2) = i", (0, 1, 0, 1)),
            (-1, 0, "e^(i·π) = -1", (1, 0, 0, 1)),
            (0, -1, "e^(i·3π/2) = -i", (0, 1, 0, 1))
        ]
        
        for x, z, label_text, color in key_points:
            # Create marker
            marker = self.loader.loadModel("models/misc/sphere")  # Updated path
            marker.setScale(0.1, 0.1, 0.1)
            marker.setPos(x, 0, z)
            marker.setColor(*color)
            marker.reparentTo(self.render)
            
            # Create label
            text = TextNode(f'point_{x}_{z}')
            text.setText(label_text)
            text.setTextColor(*color)
            text_np = self.render.attachNewNode(text)
            text_np.setPos(x * 1.2, 0, z * 1.2)
            text_np.setScale(0.2)
            text_np.setBillboardPointEye()
    
    def update_exponential(self, task):
        # Update the phase slowly
        self.phase += 0.01
        if self.phase > 2 * math.pi:
            self.phase = 0
        
        # Phase limited to 0 to π for Euler's equation visualization
        t = min(self.phase, math.pi)
        
        # Calculate position on the unit circle
        x = math.cos(t)
        z = math.sin(t)
        
        # Update the marker position
        self.angle_marker.setPos(x, 0, z)
        
        # Update the vector
        self.vector.removeNode()
        self.vector_line = LineSegs()
        self.vector_line.setThickness(4)
        self.vector_line.setColor(1, 1, 0, 1)
        self.vector_line.moveTo(0, 0, 0)
        self.vector_line.drawTo(x, 0, z)
        node = self.vector_line.create()
        self.vector = self.render.attachNewNode(node)
        
        # Update phase indicator
        if hasattr(self, 'phase_text'):
            self.phase_text.removeNode()
        
        phase_text = TextNode('phase_text')
        phase_text.setText(f"t = {t:.2f}")
        phase_text.setTextColor(1, 1, 0, 1)
        self.phase_text = self.render.attachNewNode(phase_text)
        self.phase_text.setPos(1.5, 0, -1.5)
        self.phase_text.setScale(0.3)
        self.phase_text.setBillboardPointEye()
        
        # Update equation text
        if hasattr(self, 'equation_text'):
            self.equation_text.removeNode()
        
        euler_result = complex(math.cos(t), math.sin(t))
        eq_text = TextNode('equation_text')
        eq_text.setText(f"e^(i·{t:.2f}) = {euler_result.real:.2f} + {euler_result.imag:.2f}i")
        eq_text.setTextColor(1, 1, 0, 1)
        self.equation_text = self.render.attachNewNode(eq_text)
        self.equation_text.setPos(-3, 0, -1.5)
        self.equation_text.setScale(0.3)
        self.equation_text.setBillboardPointEye()
        
        # When t = π, highlight Euler's identity
        if abs(t - math.pi) < 0.05:
            if not hasattr(self, 'highlight_text'):
                highlight_text = TextNode('highlight_text')
                highlight_text.setText("Euler's Identity: e^(iπ) + 1 = 0")
                highlight_text.setTextColor(1, 0, 0, 1)
                self.highlight_text = self.render.attachNewNode(highlight_text)
                self.highlight_text.setPos(0, 0, 2)
                self.highlight_text.setScale(0.4)
                self.highlight_text.setBillboardPointEye()
        elif hasattr(self, 'highlight_text'):
            self.highlight_text.removeNode()
            del self.highlight_text
        
        return Task.cont
    
    def rotate_camera_task(self, task):
        # Slowly rotate the camera around the scene
        self.angle += 0.1
        radius = 15
        height = 5
        x = radius * math.sin(math.radians(self.angle))
        y = radius * math.cos(math.radians(self.angle))
        self.camera.setPos(x, y, height)
        self.camera.lookAt(0, 0, 0)
        return Task.cont

# Run the application
app = EulersEquationVisualization()
app.run()