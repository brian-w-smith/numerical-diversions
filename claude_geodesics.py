#!/usr/bin/env python

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import *
import math
import numpy as np

class CausticGeodesicApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Set up the scene
        self.setBackgroundColor(0, 0, 0, 1)  # Black background
        
        # Create the geodesic sphere using procedural generation
        self.create_geodesic_sphere()
        
        # Create a custom shader for the caustic geodesic effect
        vertex_shader = """
        #version 150
        
        in vec4 p3d_Vertex;
        uniform mat4 p3d_ModelViewProjectionMatrix;
        uniform float time;
        out vec4 position;
        out vec3 color;
        
        void main() {
            position = p3d_Vertex;
            
            // Generate dynamic color based on position and time
            float r = sin(position.x * 2.0 + time) * 0.5 + 0.5;
            float g = sin(position.y * 2.0 + time * 0.7) * 0.5 + 0.5;
            float b = sin(position.z * 2.0 + time * 0.3) * 0.5 + 0.5;
            color = vec3(r, g, b);
            
            gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
        }
        """
        
        fragment_shader = """
        #version 150
        
        in vec4 position;
        in vec3 color;
        out vec4 p3d_FragColor;
        
        void main() {
            p3d_FragColor = vec4(color, 1.0);
        }
        """
        
        # Create and apply the shader to the NodePath, not the GeomNode
        shader = Shader.make(Shader.SL_GLSL, vertex_shader, fragment_shader)
        self.sphere.setShader(shader)
        
        # Set up time uniform for shader animation
        self.time = 0.0
        self.sphere.setShaderInput("time", self.time)
        
        # Set up a task to update the shader uniform
        self.taskMgr.add(self.update_shader, "UpdateShader")
        
        # Set up camera
        self.camera.setPos(0, -5, 0)
        self.camera.lookAt(0, 0, 0)
        
        # Add camera controls
        self.accept("arrow_left", self.rotate_camera, [-5, 0])
        self.accept("arrow_right", self.rotate_camera, [5, 0])
        self.accept("arrow_up", self.rotate_camera, [0, 5])
        self.accept("arrow_down", self.rotate_camera, [0, -5])
        self.camRotation = [0, 0]
        
    def create_geodesic_sphere(self, radius=1.0, subdivisions=3):
        # Create an icosahedron as the base for our geodesic sphere
        self.sphere_node = GeomNode("GeodesicSphere")
        self.sphere = self.render.attachNewNode(self.sphere_node)
        
        # Generate geodesic vertices by subdividing an icosahedron
        vertices, lines = self.generate_geodesic_data(radius, subdivisions)
        
        # Create the geodesic visualization
        self.create_geodesic_lines(vertices, lines)
        
        # Add a subtle rotation animation
        self.sphere.hprInterval(20, (360, 360, 0)).loop()
    
    def generate_geodesic_data(self, radius, subdivisions):
        # Golden ratio used to create the icosahedron
        phi = (1 + math.sqrt(5)) / 2
        
        # Initial icosahedron vertices
        base_vertices = [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]
        
        # Normalize vertices to be on unit sphere
        vertices = []
        for v in base_vertices:
            length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            vertices.append([v[0]/length*radius, v[1]/length*radius, v[2]/length*radius])
        
        # Define the edges of the icosahedron
        edges = [
            (0, 1), (0, 5), (0, 7), (0, 10), (0, 11),
            (1, 5), (1, 7), (1, 8), (1, 9),
            (2, 3), (2, 4), (2, 6), (2, 10), (2, 11),
            (3, 4), (3, 6), (3, 8), (3, 9),
            (4, 5), (4, 9), (4, 11),
            (5, 9), (5, 11),
            (6, 7), (6, 8), (6, 10),
            (7, 8), (7, 10),
            (8, 9), (10, 11)
        ]
        
        # For each subdivision, add new vertices and edges
        for _ in range(subdivisions):
            new_edges = []
            edge_points = {}  # To keep track of midpoints
            
            # For each edge, create a new vertex at its midpoint
            for i, edge in enumerate(edges):
                v1 = vertices[edge[0]]
                v2 = vertices[edge[1]]
                
                # Calculate midpoint
                midpoint = [(v1[0] + v2[0])/2, (v1[1] + v2[1])/2, (v1[2] + v2[2])/2]
                
                # Project to sphere surface
                length = math.sqrt(midpoint[0]**2 + midpoint[1]**2 + midpoint[2]**2)
                midpoint = [midpoint[0]/length*radius, midpoint[1]/length*radius, midpoint[2]/length*radius]
                
                # Add the new vertex
                new_index = len(vertices)
                vertices.append(midpoint)
                
                # Store the index of the new vertex
                edge_points[edge] = new_index
                edge_points[(edge[1], edge[0])] = new_index
                
                # Create new edges
                new_edges.append((edge[0], new_index))
                new_edges.append((new_index, edge[1]))
            
            # Update edges for the next iteration
            edges = new_edges
        
        # Return the final set of vertices and edges
        return vertices, edges
    
    def create_geodesic_lines(self, vertices, edges):
        # Create a Geom to hold the lines
        vdata = GeomVertexData('geodesic_data', GeomVertexFormat.getV3c4(), Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')
        
        # Add vertices with colors
        for v in vertices:
            vertex.addData3f(v[0], v[1], v[2])
            # Use colorful base colors
            r = (v[0] + radius) / (2 * radius)
            g = (v[1] + radius) / (2 * radius)
            b = (v[2] + radius) / (2 * radius)
            color.addData4f(r, g, b, 1.0)
        
        # Create the lines primitive
        lines = GeomLines(Geom.UHStatic)
        
        # Add all edges
        for edge in edges:
            lines.addVertices(edge[0], edge[1])
        
        # Create the Geom and add it to the node
        geom = Geom(vdata)
        geom.addPrimitive(lines)
        self.sphere_node.addGeom(geom)
        
        # Add points at vertices for a sparkle effect
        points = GeomPoints(Geom.UHStatic)
        for i in range(len(vertices)):
            points.addVertex(i)
        
        point_geom = Geom(vdata)
        point_geom.addPrimitive(points)
        
        # Set point size using RenderModeAttrib
        render_state = RenderState.make(RenderModeAttrib.make(RenderModeAttrib.MPoint, 2))
        self.sphere_node.addGeom(point_geom, render_state)
    
    def update_shader(self, task):
        # Update the time uniform for animation
        self.time += globalClock.getDt()
        self.sphere.setShaderInput("time", self.time)
        
        # Apply caustic-like distortion based on sin waves
        scale = 1.0 + 0.05 * math.sin(self.time * 2.0)
        self.sphere.setScale(scale, scale, scale)
        
        return Task.cont
    
    def rotate_camera(self, dh, dp):
        # Update camera rotation
        self.camRotation[0] += dh
        self.camRotation[1] += dp
        
        # Clamp vertical rotation to prevent flipping
        self.camRotation[1] = max(-85, min(85, self.camRotation[1]))
        
        # Calculate new position
        distance = 5.0
        rad_h = math.radians(self.camRotation[0])
        rad_p = math.radians(self.camRotation[1])
        
        x = distance * math.sin(rad_h) * math.cos(rad_p)
        y = -distance * math.cos(rad_h) * math.cos(rad_p)
        z = distance * math.sin(rad_p)
        
        self.camera.setPos(x, y, z)
        self.camera.lookAt(0, 0, 0)

# Global variables for the script
radius = 1.0

# Run the application
app = CausticGeodesicApp()
app.run()