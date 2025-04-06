from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomVertexFormat, GeomVertexData, Geom, GeomTriangles
from panda3d.core import GeomVertexWriter, GeomNode
from panda3d.core import LVector3, LPoint3
from direct.task import Task
import sys

class MengerSponge(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Set up camera
        self.disableMouse()
        self.camera.setPos(0, -15, 0)
        self.camera.lookAt(0, 0, 0)
        
        # Create light
        self.setupLights()
        
        # Create Menger sponge
        self.menger_node = self.render.attachNewNode("menger")
        self.create_menger_sponge(3)  # Create a level-3 Menger sponge
        
        # Set up rotation task
        self.taskMgr.add(self.rotate_model, "RotateTask")
        
        # Instructions
        self.instructions = self.addInstructions(0.95, "ESC: Quit")
        self.instructions = self.addInstructions(0.90, "Arrow keys: Rotate camera")
        self.instructions = self.addInstructions(0.85, "Page Up/Down: Zoom in/out")
        
        # Set up keyboard controls
        self.accept("escape", sys.exit)
        self.accept("arrow_left", self.rotate_camera, [-10, 0])
        self.accept("arrow_right", self.rotate_camera, [10, 0])
        self.accept("arrow_up", self.rotate_camera, [0, 10])
        self.accept("arrow_down", self.rotate_camera, [0, -10])
        self.accept("page_up", self.zoom_camera, [1])
        self.accept("page_down", self.zoom_camera, [-1])

    def setupLights(self):
        from panda3d.core import AmbientLight, DirectionalLight, Vec4
        
        # Add ambient light
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)
        
        # Add directional light
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setColor(Vec4(0.8, 0.8, 0.8, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setHpr(45, -45, 0)
        self.render.setLight(directionalLightNP)

    def create_cube(self, pos, size):
        # Create a cube at the given position with the given size
        format = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData('cube', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        # Define the 8 vertices of the cube
        vertices = [
            LPoint3(pos.x - size/2, pos.y - size/2, pos.z - size/2),
            LPoint3(pos.x + size/2, pos.y - size/2, pos.z - size/2),
            LPoint3(pos.x + size/2, pos.y + size/2, pos.z - size/2),
            LPoint3(pos.x - size/2, pos.y + size/2, pos.z - size/2),
            LPoint3(pos.x - size/2, pos.y - size/2, pos.z + size/2),
            LPoint3(pos.x + size/2, pos.y - size/2, pos.z + size/2),
            LPoint3(pos.x + size/2, pos.y + size/2, pos.z + size/2),
            LPoint3(pos.x - size/2, pos.y + size/2, pos.z + size/2)
        ]
        
        # Define the 6 faces of the cube
        faces = [
            (0, 1, 2, 3),  # Bottom face
            (4, 5, 6, 7),  # Top face
            (0, 4, 7, 3),  # Left face
            (1, 5, 6, 2),  # Right face
            (0, 1, 5, 4),  # Front face
            (3, 2, 6, 7)   # Back face
        ]
        
        # Define the normals for each face
        normals = [
            LVector3(0, 0, -1),  # Bottom
            LVector3(0, 0, 1),   # Top
            LVector3(-1, 0, 0),  # Left
            LVector3(1, 0, 0),   # Right
            LVector3(0, -1, 0),  # Front
            LVector3(0, 1, 0)    # Back
        ]
        
        # Define colors for each face to make it more visually interesting
        colors = [
            (1.0, 0.0, 0.0, 1.0),  # Red
            (0.0, 1.0, 0.0, 1.0),  # Green
            (0.0, 0.0, 1.0, 1.0),  # Blue
            (1.0, 1.0, 0.0, 1.0),  # Yellow
            (1.0, 0.0, 1.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0, 1.0)   # Cyan
        ]
        
        # Create triangles for each face
        tris = GeomTriangles(Geom.UHStatic)
        
        for i, face in enumerate(faces):
            for v in face:
                vertex.addData3f(vertices[v])
                normal.addData3f(normals[i])
                color.addData4f(*colors[i])
                
            # Add two triangles for each rectangular face
            vi = i * 4  # Vertex index start for this face
            tris.addVertices(vi, vi+1, vi+2)
            tris.addVertices(vi, vi+2, vi+3)
            tris.closePrimitive()
        
        # Create the Geom object
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        # Create a GeomNode to hold the geometry
        node = GeomNode('cube')
        node.addGeom(geom)
        
        return node

    def create_menger_sponge(self, level, pos=LPoint3(0, 0, 0), size=3):
        if level == 0:
            # Base case: create a cube
            cube = self.create_cube(pos, size)
            nodepath = self.menger_node.attachNewNode(cube)
            return nodepath
        
        # Recursive case: divide into 27 sub-cubes and remove the center of each face and the center cube
        new_size = size / 3
        
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    # Skip the center cube and the center of each face
                    if ((x == 1 and y == 1) or 
                        (x == 1 and z == 1) or 
                        (y == 1 and z == 1)):
                        continue
                    
                    new_pos = LPoint3(
                        pos.x + (x - 1) * new_size,
                        pos.y + (y - 1) * new_size,
                        pos.z + (z - 1) * new_size
                    )
                    
                    self.create_menger_sponge(level - 1, new_pos, new_size)

    def rotate_model(self, task):
        # Rotate the model slowly
        self.menger_node.setH((task.time * 10) % 360)
        self.menger_node.setP((task.time * 5) % 360)
        return Task.cont

    def rotate_camera(self, h_delta, p_delta):
        # Rotate the camera around the model
        current_h = self.camera.getH()
        current_p = self.camera.getP()
        self.camera.setH(current_h + h_delta)
        self.camera.setP(current_p + p_delta)

    def zoom_camera(self, delta):
        # Zoom the camera in or out
        current_pos = self.camera.getPos()
        distance = current_pos.length()
        direction = current_pos.normalized()
        new_distance = max(5, min(25, distance - delta))
        self.camera.setPos(direction * new_distance)

    def addInstructions(self, pos, msg):
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        return OnscreenText(text=msg, style=1, fg=(1, 1, 1, 1), 
                            pos=(-1.3, pos), align=TextNode.ALeft, scale=.05)

# Run the application
app = MengerSponge()
app.run()