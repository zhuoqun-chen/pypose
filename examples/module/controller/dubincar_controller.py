import torch
from pypose.module.controller import Controller

class DubinCarController(Controller):
    def __init__(self):
        super(DubinCarController, self).__init__()

    def get_control(self, parameters, state, ref_state, feed_forward_quantity):
        x_desired, y_desired, vx_desired, vy_desired, accx_desired, accy_desired, angle_desired, angle_dot_desired, angle_ddot_desired = ref_state
        px, py, orientation, vel, w = state
        kp, kv, kori, kw = parameters

        orientation_cos = torch.cos(orientation)
        orientation_sin = torch.sin(orientation)

        # acceleration output
        acceleration = kp * (orientation_cos * (x_desired - px) + orientation_sin * (y_desired - py)) \
          + kv * (orientation_cos * (vx_desired - vel * orientation_cos) + orientation_sin * (vy_desired - vel * orientation_sin)) \
          + accx_desired * orientation_cos + accy_desired * orientation_sin
        
        err_angle = angle_desired - orientation
        orientation_ddot = kori * err_angle + kw * (angle_dot_desired - w) + angle_ddot_desired

        return torch.stack([acceleration, orientation_ddot])