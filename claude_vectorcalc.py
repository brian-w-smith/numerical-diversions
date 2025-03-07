#!/usr/bin/env python

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    NodePath, LineSegs, GeomVertexFormat, GeomVertexData, Geom, GeomLines, 
    GeomNode, Vec3, Vec4, Point3, DirectionalLight, AmbientLight, TextNode,
    TransparencyAttrib
)
from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
import numpy as np
import math

class VectorCalculusDemo(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.setBackgroundColor(0.1, 0.1, 0.1)  # Dark background for better visibility
        self.title = OnscreenText(text="Vector Calculus: Gradient, Divergence & Curl",
                                 style=1, fg=(1,1,1,1), pos=(0, 0.9), scale=0.08)
        self.instruction = OnscreenText(text="Press G for Gradient, D for Divergence, C for Curl, Space to Reset",
                                       style=1, fg=(1,1,1,1), pos=(0, 0.85), scale=0.05)
        
        # Create lights
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.7, 0.7, 0.7, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)
        
        alight = AmbientLight('alight')
        alight.setColor((0.3, 0.3, 0.3, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        # Setup camera - ZOOMED OUT
        self.disableMouse()
        self.camera.setPos(30, -30, 20)  # Increased camera distance
        self.camera.lookAt(0, 0, 0)
        
        # Add camera control task
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        
        # Create coordinate system
        self.createCoordinateSystem()
        
        # Current mode
        self.current_mode = "None"
        
        # Vector field container
        self.vector_field = self.render.attachNewNode("VectorField")
        
        # Add key event handlers
        self.accept("g", self.showGradient)
        self.accept("d", self.showDivergence)
        self.accept("c", self.showCurl)
        self.accept("space", self.resetDemo)
        
        # Function for scalar field (for gradient)
        self.scalar_function = lambda x, y, z: x**2 + y**2 + z**2
        
        # Function for vector field (for divergence and curl)
        self.vector_function = lambda x, y, z: np.array([
            y**2 - z**2,    # F_x
            2*x*y,          # F_y
            z*x             # F_z
        ])
        
        # Initial message
        self.description = OnscreenText(
            text="Press G, D, or C to visualize Gradient, Divergence, or Curl",
            style=1, fg=(1,1,1,1), pos=(0, -0.85), scale=0.05
        )
    
    def createCoordinateSystem(self):
        # Create axes lines - EXTENDED
        lines = LineSegs()
        
        # X-axis (red)
        lines.setColor(1, 0, 0, 1)
        lines.moveTo(0, 0, 0)
        lines.drawTo(15, 0, 0)  # Extended axis
        
        # Y-axis (green)
        lines.setColor(0, 1, 0, 1)
        lines.moveTo(0, 0, 0)
        lines.drawTo(0, 15, 0)  # Extended axis
        
        # Z-axis (blue)
        lines.setColor(0, 0, 1, 1)
        lines.moveTo(0, 0, 0)
        lines.drawTo(0, 0, 15)  # Extended axis
        
        node = lines.create()
        axes = self.render.attachNewNode(node)
        
        # Add axis labels
        self.makeTextLabel("X", (15.5, 0, 0), (1, 0, 0, 1))
        self.makeTextLabel("Y", (0, 15.5, 0), (0, 1, 0, 1))
        self.makeTextLabel("Z", (0, 0, 15.5), (0, 0, 1, 1))
        
        # Add a reference grid on XY plane
        self.createGrid()
    
    def makeTextLabel(self, text, pos, color):
        textNode = TextNode('label')
        textNode.setText(text)
        textNode.setTextColor(color)
        textNodePath = self.render.attachNewNode(textNode)
        textNodePath.setPos(pos)
        textNodePath.setScale(0.8)  # Larger text
        textNodePath.setBillboardPointEye()
    
    def createGrid(self):
        # Create a grid on the XY plane
        lines = LineSegs()
        lines.setColor(0.5, 0.5, 0.5, 0.3)
        
        # Grid size and spacing - EXTENDED
        size = 15  # Larger grid
        step = 2   # Wider spacing
        
        # Draw grid lines
        for i in range(-size, size + 1, step):
            lines.moveTo(i, -size, 0)
            lines.drawTo(i, size, 0)
            lines.moveTo(-size, i, 0)
            lines.drawTo(size, i, 0)
        
        node = lines.create()
        grid = self.render.attachNewNode(node)
        grid.setTransparency(TransparencyAttrib.MAlpha)
    
    def spinCameraTask(self, task):
        # Slowly rotate the camera around the center
        angle_degrees = task.time * 8.0  # slower rotation speed
        angle_radians = angle_degrees * (math.pi / 180.0)
        
        radius = 40.0  # Increased orbital radius
        self.camera.setPos(
            radius * math.sin(angle_radians),
            -radius * math.cos(angle_radians),
            16.0  # Higher camera position
        )
        self.camera.lookAt(0, 0, 0)
        
        return Task.cont
    
    def resetDemo(self):
        self.vector_field.removeNode()
        self.vector_field = self.render.attachNewNode("VectorField")
        self.current_mode = "None"
        self.description.setText("Press G, D, or C to visualize Gradient, Divergence, or Curl")
    
    def createArrow(self, origin, direction, color=(1,1,1,1), scale=1.0):
        # Create an arrow to represent a vector
        if np.linalg.norm(direction) < 0.01:  # Skip very small vectors
            return None
        
        lines = LineSegs()
        lines.setColor(*color)
        
        # Normalize direction for consistent arrow heads
        dir_norm = direction / np.linalg.norm(direction)
        
        # Arrow shaft
        lines.moveTo(*origin)
        end_point = origin + direction
        lines.drawTo(*end_point)
        
        # Arrow head (simple implementation)
        head_size = 0.3 * scale  # Larger arrow heads
        
        # Create an orthogonal vector for arrow head
        if abs(dir_norm[0]) < 0.9:
            ortho = np.cross(dir_norm, np.array([1, 0, 0]))
        else:
            ortho = np.cross(dir_norm, np.array([0, 1, 0]))
        
        ortho = ortho / np.linalg.norm(ortho) * head_size
        
        # Draw arrow head
        head_point1 = end_point - dir_norm * head_size + ortho
        head_point2 = end_point - dir_norm * head_size - ortho
        
        lines.moveTo(*end_point)
        lines.drawTo(*head_point1)
        lines.moveTo(*end_point)
        lines.drawTo(*head_point2)
        
        node = lines.create()
        arrow_np = NodePath(node)
        
        return arrow_np
    
    def showGradient(self):
        self.resetDemo()
        self.current_mode = "Gradient"
        
        # Update description
        self.description.setText("Gradient (∇f): Shows direction of greatest increase of a scalar field f(x,y,z) = x² + y² + z²")
        
        # Create a grid of points to evaluate gradient - EXPANDED
        grid_points = []
        for x in np.linspace(-8, 8, 6):  # More spread out points
            for y in np.linspace(-8, 8, 6):
                for z in np.linspace(-8, 8, 6):
                    if x**2 + y**2 + z**2 <= 64:  # Limit to sphere for cleaner visualization
                        grid_points.append(np.array([x, y, z]))
        
        # Draw the scalar field as a colored sphere
        for r in np.linspace(2, 8, 4):  # Larger spheres
            self.drawContourSphere(r)
        
        # Calculate and visualize the gradient at each point
        for point in grid_points:
            x, y, z = point
            
            # Calculate gradient at this point
            gradient = np.array([
                2*x,  # df/dx
                2*y,  # df/dy
                2*z   # df/dz
            ])
            
            # Get scalar field value for coloring
            value = self.scalar_function(x, y, z)
            
            # Color depends on scalar field value
            color_intensity = min(1.0, value / 100.0)  # Adjusted for larger values
            color = (1-color_intensity, 0, color_intensity, 1)
            
            # Draw the gradient vector - LARGER
            arrow = self.createArrow(point, gradient * 0.3, color, scale=1.2)
            if arrow:
                arrow.reparentTo(self.vector_field)
    
    def drawContourSphere(self, radius):
        # Draw a sphere to represent a contour of the scalar field
        from panda3d.core import CardMaker
        
        # Create a sphere
        sphere = self.loader.loadModel("models/misc/sphere")
        sphere.setScale(radius)
        
        # Set color based on radius (value of scalar field)
        color_intensity = min(1.0, radius**2 / 100.0)  # Adjusted for larger values
        sphere.setColor(1-color_intensity, 0, color_intensity, 0.1)
        
        sphere.setTransparency(TransparencyAttrib.MAlpha)
        sphere.reparentTo(self.vector_field)
    
    def showDivergence(self):
        self.resetDemo()
        self.current_mode = "Divergence"
        
        # Update description
        self.description.setText("Divergence (∇·F): Shows outflow of vector field F(x,y,z) = [y² - z², 2xy, zx]")
        
        # Create a grid of points to evaluate divergence - EXPANDED
        grid_points = []
        for x in np.linspace(-8, 8, 6):  # More spread out points
            for y in np.linspace(-8, 8, 6):
                for z in np.linspace(-8, 8, 6):
                    if x**2 + y**2 + z**2 <= 64:  # Limit to sphere for cleaner visualization
                        grid_points.append(np.array([x, y, z]))
        
        # Calculate and visualize the vector field and its divergence
        for point in grid_points:
            x, y, z = point
            
            # Get vector field at this point
            vector = self.vector_function(x, y, z)
            
            # Calculate divergence at this point
            divergence = 2*y + z
            
            # Color based on divergence (red for positive, blue for negative)
            if divergence > 0:
                color = (min(1.0, divergence/8.0), 0, 0, 1)  # Adjusted for larger values
            else:
                color = (0, 0, min(1.0, abs(divergence)/8.0), 1)
            
            # Draw the vector field - LARGER
            arrow = self.createArrow(point, vector * 0.25, color, scale=1.2)
            if arrow:
                arrow.reparentTo(self.vector_field)
            
            # Draw a sphere to represent divergence
            sphere = self.loader.loadModel("models/misc/sphere")
            scale = abs(divergence) * 0.25  # Larger spheres
            if scale > 0.01:  # Only show significant divergence
                sphere.setScale(scale)
                sphere.setPos(*point)
                sphere.setColor(*color)
                sphere.reparentTo(self.vector_field)
    
    def showCurl(self):
        self.resetDemo()
        self.current_mode = "Curl"
        
        # Update description
        self.description.setText("Curl (∇×F): Shows rotation of vector field F(x,y,z) = [y² - z², 2xy, zx]")
        
        # Create a grid of points to evaluate curl - EXPANDED
        grid_points = []
        for x in np.linspace(-8, 8, 6):  # More spread out points
            for y in np.linspace(-8, 8, 6):
                for z in np.linspace(-8, 8, 6):
                    if x**2 + y**2 + z**2 <= 64:  # Limit to sphere for cleaner visualization
                        grid_points.append(np.array([x, y, z]))
        
        # Calculate and visualize the vector field and its curl
        for point in grid_points:
            x, y, z = point
            
            # Get vector field at this point
            vector = self.vector_function(x, y, z)
            
            # Calculate curl at this point
            curl = np.array([
                x,            # dF_z/dy - dF_y/dz = x
                -2*z,         # dF_x/dz - dF_z/dx = -2z 
                -2*x          # dF_y/dx - dF_x/dy = -2x
            ])
            
            # Draw the vector field (light gray) - LARGER
            arrow = self.createArrow(point, vector * 0.15, (0.7, 0.7, 0.7, 0.5), scale=1.0)
            if arrow:
                arrow.reparentTo(self.vector_field)
            
            # Draw the curl vector (magenta) - LARGER
            curl_magnitude = np.linalg.norm(curl)
            if curl_magnitude > 0.01:  # Only show significant curl
                curl_arrow = self.createArrow(point, curl * 0.25, (1, 0, 1, 1), scale=1.8)
                if curl_arrow:
                    curl_arrow.reparentTo(self.vector_field)

app = VectorCalculusDemo()
app.run()