import sys
import math
import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter
from panda3d.core import GeomNode, NodePath
from panda3d.core import LVector3, LVector4
from panda3d.core import PerspectiveLens, AmbientLight, DirectionalLight
from panda3d.core import TextNode, TransparencyAttrib
from direct.gui.OnscreenText import OnscreenText


class Surface3DApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Set up the camera
        self.disableMouse()
        self.camera.setPos(0, -15, 5)
        self.camera.lookAt(0, 0, 0)
        
        # Add title text
        self.title = OnscreenText(
            text="Complex 3D Surface Visualization",
            style=1, fg=(1, 1, 1, 1), pos=(0, 0.9),
            scale=0.07
        )
        
        # Add instruction text
        self.instructions = OnscreenText(
            text="Mouse: Look around | WASD: Move | Q/E: Up/Down",
            style=1, fg=(1, 1, 1, 1), pos=(0, 0.85),
            scale=0.05
        )
        
        # Set up lighting
        self.setup_lights()
        
        # Generate and render the 3D surface
        surface = self.generate_surface()
        surface.reparentTo(self.render)
        
        # Set up the movement controls
        self.setup_controls()
        
        # Add a task to rotate the surface slowly
        self.taskMgr.add(self.rotate_surface_task, "RotateSurfaceTask")

    def setup_lights(self):
        # Create ambient light
        amblight = AmbientLight("ambient")
        amblight.setColor((0.3, 0.3, 0.3, 1))
        amblightNP = self.render.attachNewNode(amblight)
        self.render.setLight(amblightNP)
        
        # Create directional lights from different angles for better shading
        for i, pos in enumerate([(15, -15, 15), (-15, -15, 15), (0, 15, 15)]):
            dirlight = DirectionalLight(f"directional{i}")
            dirlight.setColor((0.8, 0.8, 0.8, 1))
            dirlightNP = self.render.attachNewNode(dirlight)
            dirlightNP.setPos(*pos)
            dirlightNP.lookAt(0, 0, 0)
            self.render.setLight(dirlightNP)

    def generate_surface(self):
        # Parameters for the surface
        width, depth = 6, 6  # Total width and depth of the surface
        segments = 100  # Number of segments in each direction
        
        # Create the vertex format
        format = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData('surface', format, Geom.UHStatic)
        
        # Create writers for vertex position, normal, and color
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        # Create the mesh grid
        x = np.linspace(-width/2, width/2, segments)
        y = np.linspace(-depth/2, depth/2, segments)
        X, Y = np.meshgrid(x, y)
        
        # Create the complex surface
        R = np.sqrt(X**2 + Y**2)
        Z1 = np.sin(R) / (R + 0.001) * 0.5
        Z2 = np.sin(X*2) * np.cos(Y*2) * 0.25
        Z3 = np.exp(-(X**2 + Y**2)/4) * np.cos(R*4) * 0.5
        Z = Z1 + Z2 + Z3
        
        # Helper function to calculate normal at a specific point
        def calculate_normal(i, j):
            # Calculate partial derivatives for normal
            if i > 0 and i < segments-1 and j > 0 and j < segments-1:
                dx = (Z[j, i+1] - Z[j, i-1]) / (x[i+1] - x[i-1])
                dy = (Z[j+1, i] - Z[j-1, i]) / (y[j+1] - y[j-1])
            else:
                dx = dy = 0
            
            # Normal vector is (-dx, -dy, 1) normalized
            normal_vec = LVector3(-dx, -dy, 1)
            normal_vec.normalize()
            return normal_vec
        
        # Helper function to map values to colors (HSV-based coloring)
        def get_color(z_val):
            # Map z from [-1, 1] to [0, 1] for hue
            hue = (z_val + 1) / 2
            
            # Convert HSV to RGB (simplified conversion)
            h = hue * 6
            i = int(h)
            f = h - i
            
            q = 1 - f
            t = f
            
            if i % 6 == 0:
                r, g, b = 1, t, 0
            elif i % 6 == 1:
                r, g, b = q, 1, 0
            elif i % 6 == 2:
                r, g, b = 0, 1, t
            elif i % 6 == 3:
                r, g, b = 0, q, 1
            elif i % 6 == 4:
                r, g, b = t, 0, 1
            else:
                r, g, b = 1, 0, q
            
            return (r, g, b, 1)
        
        # Add vertices with position, normal, and color
        for j in range(segments):
            for i in range(segments):
                # Vertex position
                vertex.addData3(X[j, i], Y[j, i], Z[j, i])
                
                # Vertex normal
                n = calculate_normal(i, j)
                normal.addData3(n.x, n.y, n.z)
                
                # Vertex color based on height
                c = get_color(Z[j, i])
                color.addData4(c[0], c[1], c[2], c[3])
        
        # Create triangles
        tris = GeomTriangles(Geom.UHStatic)
        
        for j in range(segments - 1):
            for i in range(segments - 1):
                # Calculate vertex indices
                v0 = j * segments + i
                v1 = j * segments + i + 1
                v2 = (j + 1) * segments + i
                v3 = (j + 1) * segments + i + 1
                
                # Add two triangles for each grid cell
                tris.addVertices(v0, v1, v2)
                tris.closePrimitive()
                
                tris.addVertices(v2, v1, v3)
                tris.closePrimitive()
        
        # Create the geometry
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        # Create a node and attach the geometry
        node = GeomNode('surface')
        node.addGeom(geom)
        
        # Create and return the NodePath
        surface = NodePath(node)
        surface.setTwoSided(True)  # Make the surface visible from both sides
        
        return surface

    def setup_controls(self):
        # Set up keyboard controls for WASD movement
        self.keyMap = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "up": False,
            "down": False
        }
        
        # Accept keyboard input
        self.accept("w", self.update_key_map, ["forward", True])
        self.accept("w-up", self.update_key_map, ["forward", False])
        self.accept("s", self.update_key_map, ["backward", True])
        self.accept("s-up", self.update_key_map, ["backward", False])
        self.accept("a", self.update_key_map, ["left", True])
        self.accept("a-up", self.update_key_map, ["left", False])
        self.accept("d", self.update_key_map, ["right", True])
        self.accept("d-up", self.update_key_map, ["right", False])
        self.accept("q", self.update_key_map, ["up", True])
        self.accept("q-up", self.update_key_map, ["up", False])
        self.accept("e", self.update_key_map, ["down", True])
        self.accept("e-up", self.update_key_map, ["down", False])
        self.accept("escape", sys.exit)
        
        # Add the movement task
        self.taskMgr.add(self.movement_task, "MovementTask")
        
        # Set up mouse look
        self.mouseLookEnabled = True
        self.recenterMouse()
        self.taskMgr.add(self.mouse_look_task, "MouseLookTask")

    def update_key_map(self, key, value):
        self.keyMap[key] = value

    def movement_task(self, task):
        # Get elapsed time
        dt = globalClock.getDt()
        
        # Move the camera based on key presses
        if self.keyMap["forward"]:
            self.camera.setPos(self.camera, 0, 5 * dt, 0)
        if self.keyMap["backward"]:
            self.camera.setPos(self.camera, 0, -5 * dt, 0)
        if self.keyMap["left"]:
            self.camera.setPos(self.camera, -5 * dt, 0, 0)
        if self.keyMap["right"]:
            self.camera.setPos(self.camera, 5 * dt, 0, 0)
        if self.keyMap["up"]:
            self.camera.setPos(self.camera, 0, 0, 5 * dt)
        if self.keyMap["down"]:
            self.camera.setPos(self.camera, 0, 0, -5 * dt)
        
        return Task.cont

    def recenterMouse(self):
        self.win.movePointer(0, self.win.getXSize() // 2, self.win.getYSize() // 2)

    def mouse_look_task(self, task):
        if not self.mouseLookEnabled:
            return Task.cont
        
        # Get the mouse position
        md = self.win.getPointer(0)
        x = md.getX()
        y = md.getY()
        
        # Get window center
        window_center_x = self.win.getXSize() // 2
        window_center_y = self.win.getYSize() // 2
        
        # Calculate mouse movement from center
        dx = (x - window_center_x) * 0.1
        dy = (y - window_center_y) * 0.1
        
        # Adjust camera heading and pitch
        current_h = self.camera.getH()
        current_p = self.camera.getP()
        
        self.camera.setH(current_h - dx)
        self.camera.setP(current_p - dy)
        
        # Recenter mouse
        self.recenterMouse()
        
        return Task.cont

    def rotate_surface_task(self, task):
        # Slowly rotate the surface
        surface = self.render.find("**/surface")
        if not surface.isEmpty():
            surface.setH(task.time * 10)  # Rotate 10 degrees per second
        
        return Task.cont


app = Surface3DApp()
app.run()