from vector import Vector3d

class WayPoint:
  def __init__(self, px, py, pz=0, timestamp=0):
    self.position = Vector3d()
    self.vel = Vector3d()
    self.acc = Vector3d()
    self.ts = 0.0
    self.position.x = px
    self.position.y = py
    self.position.z = pz
    self.ts = timestamp

  def setTime(self, timestamp):
    self.ts = timestamp