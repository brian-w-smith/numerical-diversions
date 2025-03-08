#!/usr/bin/env python

from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import *
from panda3d.core import *
from direct.task import Task
import math
import numpy as np

class MultivariableCalculusSimulator(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Set up the camera
        self.disableMouse()
        self.camera.setPos(0, -15, 8)
        self.camera.lookAt(0, 0, 0)
        
        # Add ambient light
        alight = AmbientLight('alight')
        alight.setColor((0.3, 0.3, 0.3, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        # Add directional light
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)
        
        # Set up the title
        self.title = OnscreenText(text="Multivariable Calculus Simulator",
                                 style=1, fg=(1, 1, 1, 1),
                                 pos=(0, 0.9), scale=0.07)
        
        # Set up the instructions
        self.instructions = OnscreenText(text="Use 1-5 keys to switch between demonstrations\n"
                                        "Use arrow keys to rotate the view\n"
                                        "Use +/- keys to zoom in/out\n",
                                        style=1, fg=(1, 1, 1, 1),
                                        pos=(-1.3, 0.85), align=TextNode.ALeft, scale=0.05)
        
        # Set up the description text
        self.description = OnscreenText(text="",
                                      style=1, fg=(1, 1, 1, 1),
                                      pos=(-1.3, 0.7), align=TextNode.ALeft, scale=0.05,
                                      mayChange=True)
        
        # Set up controls
        self.accept("1", self.set_demo, [1])
        self.accept("2", self.set_demo, [2])
        self.accept("3", self.set_demo, [3])
        self.accept("4", self.set_demo, [4])
        self.accept("5", self.set_demo, [5])
        self.accept("arrow_left", self.rotate_camera, [-5, 0])
        self.accept("arrow_right", self.rotate_camera, [5, 0])
        self.accept("arrow_up", self.rotate_camera, [0, 5])
        self.accept("arrow_down", self.rotate_camera, [0, -5])
        self.accept("+", self.zoom_camera, [-1])
        self.accept("-", self.zoom_camera, [1])
        
        # Camera control variables
        self.cam_distance = 15
        self.cam_h = 0
        self.cam_p = -30
        
        # Demo variables
        self.current_demo = 0
        self.function_node = self.render.attachNewNode("function")
        self.gradient_node = self.render.attachNewNode("gradient")
        self.path_node = self.render.attachNewNode("path")
        self.vector_field_node = self.render.attachNewNode("vector_field")
        self.parameterized_surface_node = self.render.attachNewNode("param_surface")
        
        # Set up the first demo
        self.set_demo(1)
    
    def rotate_camera(self, h, p):
        self.cam_h += h
        self.cam_p += p
        self.cam_p = max(-85, min(85, self.cam_p))  # Clamp pitch
        self.update_camera()
    
    def zoom_camera(self, change):
        self.cam_distance += change
        self.cam_distance = max(5, min(30, self.cam_distance))  # Clamp distance
        self.update_camera()
    
    def update_camera(self):
        # Convert spherical coordinates to Cartesian
        rad_h = math.radians(self.cam_h)
        rad_p = math.radians(self.cam_p)
        x = self.cam_distance * math.sin(rad_h) * math.cos(rad_p)
        y = -self.cam_distance * math.cos(rad_h) * math.cos(rad_p)
        z = self.cam_distance * math.sin(rad_p)
        
        self.camera.setPos(x, y, z)
        self.camera.lookAt(0, 0, 0)
    
    def set_demo(self, demo_num):
        # Clear previous demo
        self.function_node.removeNode()
        self.gradient_node.removeNode()
        self.path_node.removeNode()
        self.vector_field_node.removeNode()
        self.parameterized_surface_node.removeNode()
        
        # Create new nodes
        self.function_node = self.render.attachNewNode("function")
        self.gradient_node = self.render.attachNewNode("gradient")
        self.path_node = self.render.attachNewNode("path")
        self.vector_field_node = self.render.attachNewNode("vector_field")
        self.parameterized_surface_node = self.render.attachNewNode("param_surface")
        
        self.current_demo = demo_num
        
        if demo_num == 1:
            self.create_3d_function()
            self.description.setText("Demo 1: 3D Function Visualization\n\n"
                                   "f(x,y) = sin(x) * cos(y)\n\n"
                                   "This demonstrates a scalar-valued function of two variables,\n"
                                   "represented as a surface in 3D space.")
        elif demo_num == 2:
            self.create_3d_function()
            self.create_gradient_field()
            self.description.setText("Demo 2: Gradient Vector Field\n\n"
                                   "∇f(x,y) = (∂f/∂x, ∂f/∂y)\n\n"
                                   "The gradient vectors (shown in red) point in the\n"
                                   "direction of steepest increase of the function.")
        elif demo_num == 3:
            self.create_3d_function()
            self.create_path_integral()
            self.description.setText("Demo 3: Path Integral Demonstration\n\n"
                                   "∫_C f(x,y) ds\n\n"
                                   "The green path shows a parameterized curve C.\n"
                                   "The path integral accumulates the value of the function\n"
                                   "along this curve.")
        elif demo_num == 4:
            self.create_vector_field()
            self.description.setText("Demo 4: Vector Field and Curl\n\n"
                                   "F(x,y) = (-y, x)\n\n"
                                   "This vector field represents a rotational field.\n"
                                   "The curl (rotation) is constant and points in the z direction.")
        elif demo_num == 5:
            self.create_parameterized_surface()
            self.description.setText("Demo 5: Parameterized Surface and Flux\n\n"
                                   "r(u,v) = (cos(u)sin(v), sin(u)sin(v), cos(v))\n\n"
                                   "This demonstrates a sphere parameterized by\n"
                                   "spherical coordinates. The surface normal vectors\n"
                                   "are shown in blue.")
    
    def create_3d_function(self):
        # Create a 3D surface representing f(x,y) = sin(x) * cos(y)
        maker = CardMaker('plane')
        maker.setFrame(-4, 4, -4, 4)
        plane = self.function_node.attachNewNode(maker.generate())
        plane.setPos(0, 0, 0)
        plane.setP(-90)
        
        # Create a shader to visualize the function
        vertex_shader = """
        #version 150
        
        uniform mat4 p3d_ModelViewProjectionMatrix;
        in vec4 p3d_Vertex;
        in vec2 p3d_MultiTexCoord0;
        out vec2 texcoord;
        
        void main() {
            gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
            texcoord = p3d_MultiTexCoord0;
        }
        """
        
        fragment_shader = """
        #version 150
        
        in vec2 texcoord;
        out vec4 color;
        
        void main() {
            // Scale texture coordinates to the range [-4, 4]
            float x = (texcoord.x * 8.0) - 4.0;
            float y = (texcoord.y * 8.0) - 4.0;
            
            // Calculate function value: f(x,y) = sin(x) * cos(y)
            float z = sin(x) * cos(y);
            
            // Normalize to [0,1] for coloring
            float normalized_z = (z + 1.0) / 2.0;
            
            // Create a color gradient based on height
            vec3 color_low = vec3(0.0, 0.2, 0.7);  // Blue for low values
            vec3 color_high = vec3(1.0, 0.0, 0.0);  // Red for high values
            vec3 final_color = mix(color_low, color_high, normalized_z);
            
            color = vec4(final_color, 1.0);
        }
        """
        
        geometry_shader = """
        #version 150
        
        layout(triangles) in;
        layout(triangle_strip, max_vertices=3) out;
        
        in vec2 texcoord[];
        out vec2 g_texcoord;
        out vec3 g_normal;
        
        uniform mat4 p3d_ModelViewProjectionMatrix;
        
        void main() {
            // Calculate the function values for each vertex
            float x0 = (texcoord[0].x * 8.0) - 4.0;
            float y0 = (texcoord[0].y * 8.0) - 4.0;
            float z0 = sin(x0) * cos(y0);
            
            float x1 = (texcoord[1].x * 8.0) - 4.0;
            float y1 = (texcoord[1].y * 8.0) - 4.0;
            float z1 = sin(x1) * cos(y1);
            
            float x2 = (texcoord[2].x * 8.0) - 4.0;
            float y2 = (texcoord[2].y * 8.0) - 4.0;
            float z2 = sin(x2) * cos(y2);
            
            // Create the deformed triangle
            vec4 v0 = vec4(x0, y0, z0, 1.0);
            vec4 v1 = vec4(x1, y1, z1, 1.0);
            vec4 v2 = vec4(x2, y2, z2, 1.0);
            
            // Calculate the normal of the triangle
            vec3 edge1 = vec3(v1 - v0);
            vec3 edge2 = vec3(v2 - v0);
            vec3 normal = normalize(cross(edge1, edge2));
            
            // Output the triangle with the calculated normal
            gl_Position = p3d_ModelViewProjectionMatrix * v0;
            g_texcoord = texcoord[0];
            g_normal = normal;
            EmitVertex();
            
            gl_Position = p3d_ModelViewProjectionMatrix * v1;
            g_texcoord = texcoord[1];
            g_normal = normal;
            EmitVertex();
            
            gl_Position = p3d_ModelViewProjectionMatrix * v2;
            g_texcoord = texcoord[2];
            g_normal = normal;
            EmitVertex();
            
            EndPrimitive();
        }
        """
        
        shader = Shader.make(Shader.SL_GLSL, vertex_shader, fragment_shader, geometry_shader)
        plane.setShader(shader)
        
        # Add a wireframe to help visualize the surface
        # Create a grid of lines instead of using a model
        self.create_grid(-4, 4, -4, 4, 8, 8)
        
        # Add coordinate axes
        self.create_axes()
    
    def create_grid(self, x_min, x_max, y_min, y_max, x_divisions, y_divisions):
        line_segs = LineSegs()
        line_segs.setColor(0.5, 0.5, 0.5, 0.5)
        line_segs.setThickness(1)
        
        # Create horizontal lines
        for i in range(y_divisions + 1):
            y = y_min + (y_max - y_min) * i / y_divisions
            line_segs.moveTo(x_min, y, -2)
            line_segs.drawTo(x_max, y, -2)
        
        # Create vertical lines
        for i in range(x_divisions + 1):
            x = x_min + (x_max - x_min) * i / x_divisions
            line_segs.moveTo(x, y_min, -2)
            line_segs.drawTo(x, y_max, -2)
        
        grid_node = self.function_node.attachNewNode(line_segs.create())
        grid_node.setBin("fixed", 1)
    
    def create_gradient_field(self):
        # Add gradient vectors to the surface
        for x in np.linspace(-3.5, 3.5, 10):
            for y in np.linspace(-3.5, 3.5, 10):
                # Calculate function value and gradient
                z = math.sin(x) * math.cos(y)
                
                # Partial derivatives
                dx = math.cos(x) * math.cos(y)
                dy = -math.sin(x) * math.sin(y)
                
                # Create a vector to represent the gradient
                self.create_vector((x, y, z), (dx, dy, 0), scale=0.2, color=(1, 0, 0, 1), parent=self.gradient_node)
    
    def create_path_integral(self):
        # Create a parameterized path C on the surface
        line_segs = LineSegs()
        line_segs.setColor(0, 1, 0, 1)
        line_segs.setThickness(3)
        
        # Parametric path: spiral
        t_values = np.linspace(0, 6*math.pi, 100)
        
        points = []
        for t in t_values:
            # Parameterized path: spiral from center outward
            radius = t / (6*math.pi) * 3
            x = radius * math.cos(t)
            y = radius * math.sin(t)
            z = math.sin(x) * math.cos(y)
            points.append((x, y, z))
        
        # Create the line segments for the path
        for i in range(len(points) - 1):
            line_segs.moveTo(points[i][0], points[i][1], points[i][2])
            line_segs.drawTo(points[i+1][0], points[i+1][1], points[i+1][2])
        
        path_node = self.path_node.attachNewNode(line_segs.create())
        path_node.setBin("fixed", 1)
    
    def create_vector_field(self):
        # Create a vector field F(x,y) = (-y, x, 0)
        for x in np.linspace(-3.5, 3.5, 8):
            for y in np.linspace(-3.5, 3.5, 8):
                # Vector field components
                vx = -y
                vy = x
                vz = 0
                
                magnitude = math.sqrt(vx*vx + vy*vy + vz*vz)
                normalized_magnitude = min(1.0, magnitude / 3.5)
                
                # Color based on magnitude
                color = (0.2, 0.2, 0.8 + 0.2 * normalized_magnitude)
                
                self.create_vector((x, y, 0), (vx, vy, vz), scale=0.3, color=color, parent=self.vector_field_node)
        
        # Add coordinate axes
        self.create_axes()
    
    def create_parameterized_surface(self):
        # Create a parameterized sphere using points and lines
        line_segs = LineSegs()
        line_segs.setThickness(1)
        
        # Create latitude lines
        for v in np.linspace(0, math.pi, 10):
            line_segs.setColor(0.7, 0.7, 0.7, 1)
            for u in np.linspace(0, 2*math.pi, 40):
                x = 2 * math.cos(u) * math.sin(v)
                y = 2 * math.sin(u) * math.sin(v)
                z = 2 * math.cos(v)
                
                if u == 0:
                    line_segs.moveTo(x, y, z)
                else:
                    line_segs.drawTo(x, y, z)
            # Close the loop
            x = 2 * math.cos(0) * math.sin(v)
            y = 2 * math.sin(0) * math.sin(v)
            z = 2 * math.cos(v)
            line_segs.drawTo(x, y, z)
        
        # Create longitude lines
        for u in np.linspace(0, 2*math.pi, 20, endpoint=False):
            line_segs.setColor(0.7, 0.7, 0.7, 1)
            for v in np.linspace(0, math.pi, 20):
                x = 2 * math.cos(u) * math.sin(v)
                y = 2 * math.sin(u) * math.sin(v)
                z = 2 * math.cos(v)
                
                if v == 0:
                    line_segs.moveTo(x, y, z)
                else:
                    line_segs.drawTo(x, y, z)
        
        sphere_node = self.parameterized_surface_node.attachNewNode(line_segs.create())
        
        # Create surface normals at sample points
        for u in np.linspace(0, 2*math.pi, 12, endpoint=False):
            for v in np.linspace(0, math.pi, 6):
                # Spherical coordinates to Cartesian
                x = 2 * math.cos(u) * math.sin(v)
                y = 2 * math.sin(u) * math.sin(v)
                z = 2 * math.cos(v)
                
                # Surface normal at this point is just the normalized position for a sphere
                normal = (x/2, y/2, z/2)
                
                self.create_vector((x, y, z), normal, scale=0.3, color=(0, 0, 1, 1), parent=self.parameterized_surface_node)
        
        # Add coordinate axes
        self.create_axes()
    
    def create_vector(self, pos, direction, scale=1.0, color=(1, 1, 1, 1), parent=None):
        if parent is None:
            parent = self.render
            
        # Normalize direction
        length = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        if length < 0.001:
            length = 0.001
        
        norm_dir = (direction[0]/length, direction[1]/length, direction[2]/length)
        
        # Calculate endpoint
        end_x = pos[0] + norm_dir[0] * scale
        end_y = pos[1] + norm_dir[1] * scale
        end_z = pos[2] + norm_dir[2] * scale
        
        # Create line segment for vector
        line_segs = LineSegs()
        line_segs.setColor(*color)
        line_segs.setThickness(2)
        line_segs.moveTo(pos[0], pos[1], pos[2])
        line_segs.drawTo(end_x, end_y, end_z)
        
        vector_node = parent.attachNewNode(line_segs.create())
        
        # Create colored point at the end to represent arrow head
        cm = CardMaker('point')
        cm.setFrame(-0.05, 0.05, -0.05, 0.05)
        point = parent.attachNewNode(cm.generate())
        point.setPos(end_x, end_y, end_z)
        point.setColor(*color)
        point.setBillboardPointEye()
        
        return vector_node
    
    def create_axes(self):
        # Create x, y, z axes
        line_segs = LineSegs()
        
        # X-axis (red)
        line_segs.setColor(1, 0, 0, 1)
        line_segs.setThickness(2)
        line_segs.moveTo(0, 0, 0)
        line_segs.drawTo(5, 0, 0)
        
        # Y-axis (green)
        line_segs.setColor(0, 1, 0, 1)
        line_segs.moveTo(0, 0, 0)
        line_segs.drawTo(0, 5, 0)
        
        # Z-axis (blue)
        line_segs.setColor(0, 0, 1, 1)
        line_segs.moveTo(0, 0, 0)
        line_segs.drawTo(0, 0, 5)
        
        axes = self.render.attachNewNode(line_segs.create())
        axes.setBin("fixed", 1)
        
        # Add labels
        x_label = TextNode('x_label')
        x_label.setText("X")
        x_label_np = self.render.attachNewNode(x_label)
        x_label_np.setPos(5.2, 0, 0)
        x_label_np.setColor(1, 0, 0, 1)
        x_label_np.setBillboardPointEye()
        
        y_label = TextNode('y_label')
        y_label.setText("Y")
        y_label_np = self.render.attachNewNode(y_label)
        y_label_np.setPos(0, 5.2, 0)
        y_label_np.setColor(0, 1, 0, 1)
        y_label_np.setBillboardPointEye()
        
        z_label = TextNode('z_label')
        z_label.setText("Z")
        z_label_np = self.render.attachNewNode(z_label)
        z_label_np.setPos(0, 0, 5.2)
        z_label_np.setColor(0, 0, 1, 1)
        z_label_np.setBillboardPointEye()

# Run the simulation
app = MultivariableCalculusSimulator()
app.run()