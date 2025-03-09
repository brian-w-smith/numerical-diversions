from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, NodePath, Vec3, LPoint3f
from panda3d.core import AmbientLight, DirectionalLight
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import numpy as np
from collections import deque

class StrangeAttractor(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Set up the camera
        self.disableMouse()
        self.camera.setPos(60, -60, 20)
        self.camera.lookAt(0, 0, 0)
        
        # Attractor type - can be changed to other types
        self.attractor_type = "lorenz"  # Options: "lorenz", "aizawa", "chen", "thomas"
        
        # Parameters for various attractors
        self.params = {
            "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0/3.0},
            "aizawa": {"a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5, "e": 0.25, "f": 0.1},
            "chen": {"a": 5.0, "b": -10.0, "c": -0.38},
            "thomas": {"b": 0.208186}
        }
        
        # Initial conditions
        self.x = 0.1
        self.y = 0.1
        self.z = 0.1
        
        # Set up lighting
        self.setupLights()
        
        # Store points in a deque for the trail effect
        self.max_trail_length = 10000  # Maximum number of points to keep
        self.points = deque(maxlen=self.max_trail_length)
        self.points.append((self.x, self.y, self.z))
        
        # Create node for the attractor
        self.attractor_node = NodePath("attractor")
        self.attractor_node.reparentTo(self.render)
        
        # Scale factor for rendering (some attractors need different scaling)
        self.scale_factors = {
            "lorenz": 1.2,
            "aizawa": 15.0,
            "chen": 2.0,
            "thomas": 6.0
        }
        
        # Add a task to update the attractor
        self.taskMgr.add(self.updateAttractor, "updateAttractor")
        
        # Add a task to rotate the camera
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        
        # Add instructions
        self.instruction = self.addInstructions(0.95, f"{self.attractor_type.capitalize()} Attractor with Trail - Press ESC to exit")
    
    def setupLights(self):
        # Add ambient light
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((0.3, 0.3, 0.3, 1.0))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)
        
        # Add directional light
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setColor((0.8, 0.8, 0.8, 1.0))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setHpr(0, -45, 0)
        self.render.setLight(directionalLightNP)
    
    def addInstructions(self, pos, msg):
        return OnscreenText(text=msg, style=1, fg=(1, 1, 1, 1), 
                            pos=(-1.3, pos), align=TextNode.ALeft, scale=.05)
    
    def calculate_next_point(self, x, y, z, dt):
        """Calculate next point based on chosen attractor type."""
        if self.attractor_type == "lorenz":
            return self.lorenz_system(x, y, z, dt)
        elif self.attractor_type == "aizawa":
            return self.aizawa_system(x, y, z, dt)
        elif self.attractor_type == "chen":
            return self.chen_system(x, y, z, dt)
        elif self.attractor_type == "thomas":
            return self.thomas_system(x, y, z, dt)
        else:
            return self.lorenz_system(x, y, z, dt)  # Default to Lorenz
    
    def lorenz_system(self, x, y, z, dt):
        """Compute the next point in the Lorenz system."""
        sigma = self.params["lorenz"]["sigma"]
        rho = self.params["lorenz"]["rho"]
        beta = self.params["lorenz"]["beta"]
        
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        # Update using Euler integration
        new_x = x + dx * dt
        new_y = y + dy * dt
        new_z = z + dz * dt
        
        return new_x, new_y, new_z
    
    def aizawa_system(self, x, y, z, dt):
        """Compute the next point in the Aizawa system."""
        a = self.params["aizawa"]["a"]
        b = self.params["aizawa"]["b"]
        c = self.params["aizawa"]["c"]
        d = self.params["aizawa"]["d"]
        e = self.params["aizawa"]["e"]
        f = self.params["aizawa"]["f"]
        
        dx = (z - b) * x - d * y
        dy = d * x + (z - b) * y
        dz = c + a * z - z**3/3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3
        
        # Update using Euler integration
        new_x = x + dx * dt
        new_y = y + dy * dt
        new_z = z + dz * dt
        
        return new_x, new_y, new_z
    
    def chen_system(self, x, y, z, dt):
        """Compute the next point in the Chen system."""
        a = self.params["chen"]["a"]
        b = self.params["chen"]["b"]
        c = self.params["chen"]["c"]
        
        dx = a * x - y * z
        dy = b * y + x * z
        dz = c * z + x * y / 3
        
        # Update using Euler integration
        new_x = x + dx * dt
        new_y = y + dy * dt
        new_z = z + dz * dt
        
        return new_x, new_y, new_z
    
    def thomas_system(self, x, y, z, dt):
        """Compute the next point in the Thomas system."""
        b = self.params["thomas"]["b"]
        
        dx = -b * x + np.sin(y)
        dy = -b * y + np.sin(z)
        dz = -b * z + np.sin(x)
        
        # Update using Euler integration
        new_x = x + dx * dt
        new_y = y + dy * dt
        new_z = z + dz * dt
        
        return new_x, new_y, new_z
    
    def updateAttractor(self, task):
        """Update the attractor by adding new points."""
        # Time step
        dt = 0.005
        
        # Calculate several points per frame to make the attractor grow faster
        for _ in range(10):
            # Calculate the next point
            new_x, new_y, new_z = self.calculate_next_point(self.x, self.y, self.z, dt)
            
            # Add the point to our trail
            self.points.append((new_x, new_y, new_z))
            
            # Update the current position
            self.x, self.y, self.z = new_x, new_y, new_z
        
        # Redraw the entire trail with color gradient
        self.drawTrail()
        
        return Task.cont
    
    def drawTrail(self):
        """Draw the attractor trail with a color gradient."""
        # Remove the previous lines
        self.attractor_node.removeNode()
        self.attractor_node = NodePath("attractor")
        self.attractor_node.reparentTo(self.render)
        
        # Create multiple line segments with different colors for the trail effect
        num_segments = 10  # Number of color segments
        points_per_segment = len(self.points) // num_segments
        
        if points_per_segment < 2:  # Need at least 2 points to draw a line
            return
        
        for i in range(num_segments):
            # Calculate the color for this segment (newer segments are brighter)
            # Using a rainbow color effect
            hue = i / num_segments
            r, g, b = self.hsv_to_rgb(hue, 1.0, 1.0)
            
            # Create line segment
            lines = LineSegs()
            lines.setThickness(2.0)
            
            # Newer segments are more opaque
            alpha = 0.3 + 0.7 * (i / (num_segments - 1)) if num_segments > 1 else 1.0
            lines.setColor(r, g, b, alpha)
            
            # Get points for this segment
            start_idx = i * points_per_segment
            end_idx = (i + 1) * points_per_segment if i < num_segments - 1 else len(self.points)
            
            segment_points = list(self.points)[start_idx:end_idx]
            if len(segment_points) < 2:
                continue
            
            # Draw the line for this segment
            lines.moveTo(*segment_points[0])
            for point in segment_points[1:]:
                lines.drawTo(*point)
            
            # Add this segment to the attractor node
            segment_node = NodePath(lines.create())
            segment_node.reparentTo(self.attractor_node)
        
        # Scale the attractor to be more visible - use specific scale for each attractor type
        scale_factor = self.scale_factors.get(self.attractor_type, 1.0)
        self.attractor_node.setScale(scale_factor)
    
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV color to RGB."""
        if s == 0.0:
            return v, v, v
        
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i %= 6
        
        if i == 0:
            return v, t, p
        if i == 1:
            return q, v, p
        if i == 2:
            return p, v, t
        if i == 3:
            return p, q, v
        if i == 4:
            return t, p, v
        if i == 5:
            return v, p, q
    
    def spinCameraTask(self, task):
        """Rotate the camera around the attractor."""
        angle_degrees = task.time * 10.0  # Rotate at 10 degrees per second
        angle_radians = angle_degrees * (np.pi / 180.0)
        
        # Update camera position with increased radius for zoomed out view
        radius = 150.0
        self.camera.setPos(
            radius * np.sin(angle_radians),
            -radius * np.cos(angle_radians),
            20.0
        )
        self.camera.lookAt(0, 0, 20)
        
        return Task.cont

# Run the application
app = StrangeAttractor()
app.run()