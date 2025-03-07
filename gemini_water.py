from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
import math
import random

class WaterEffect(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Set up the camera
        self.camera.setPos(0, -20, 5)
        self.camera.lookAt(0, 0, 0)

        # Procedural plane creation
        self.water_plane = self.create_procedural_wireframe_plane(10, 10, 100, 100)
        self.water_plane.reparentTo(self.render)
        self.water_plane.setPos(0, 0, 0)

        # Wave parameters
        self.wave_speed = 0.5
        self.wave_height = 0.5
        self.wave_frequency = 2.0
        self.wave_phase = 0.0

        # Update task
        self.taskMgr.add(self.update_water, "update_water")

    def create_procedural_wireframe_plane(self, width, height, x_subdivisions, y_subdivisions):
        vdata = GeomVertexData('plane', GeomVertexFormat.getV3(), Geom.UHStatic)
        vwriter = GeomVertexWriter(vdata, 'vertex')
        lines = GeomLines(Geom.UHStatic)

        x_step = width / x_subdivisions
        y_step = height / y_subdivisions

        for y in range(y_subdivisions + 1):
            for x in range(x_subdivisions + 1):
                vwriter.addData3f(x * x_step - width / 2, y * y_step - height / 2, 0)

        for y in range(y_subdivisions + 1):
            for x in range(x_subdivisions):
                v1 = (y * (x_subdivisions + 1)) + x
                v2 = (y * (x_subdivisions + 1)) + x + 1
                lines.addVertices(v1, v2)

        for y in range(y_subdivisions):
            for x in range(x_subdivisions + 1):
                v1 = (y * (x_subdivisions + 1)) + x
                v2 = ((y + 1) * (x_subdivisions + 1)) + x
                lines.addVertices(v1, v2)

        geom = Geom(vdata)
        geom.addPrimitive(lines)
        node = GeomNode('plane')
        node.addGeom(geom)
        return NodePath(node)

    def update_water(self, task):
        self.wave_phase += self.wave_speed * globalClock.getDt()

        # Access the GeomNode and Geom
        geom_node = self.water_plane.node()
        original_geom = geom_node.getGeom(0)
        original_vdata = original_geom.getVertexData()

        # Create a new GeomVertexData for each frame
        new_vdata = GeomVertexData(original_vdata)
        vertices = GeomVertexReader(new_vdata, 'vertex')
        writer = GeomVertexWriter(new_vdata, 'vertex')

        for i in range(new_vdata.getNumRows()):
            x = vertices.getData3f()[0]
            y = vertices.getData3f()[1]
            z = self.wave_height * math.sin(self.wave_frequency * (x + self.wave_phase))
            writer.setData3f(x, y, z)
            vertices.setRow(i)

        # Create a new Geom with the modified vertex data
        new_geom = Geom(new_vdata)
        new_geom.addPrimitive(original_geom.getPrimitive(0))

        # Replace the original Geom with the modified copy
        geom_node.setGeom(0, new_geom)

        return task.cont

app = WaterEffect()
app.run()