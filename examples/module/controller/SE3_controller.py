import torch
from pypose.module.controller import Controller
from examples.module.controller_parameters_tuner.commons \
  import hat, vee, quaternion_2_rotation_matrix


class SE3Controller(Controller):
    def __init__(self, mass, J, e3, g = 9.81):
        super(SE3Controller, self).__init__()
        self.e3 = e3.double()
        self.g = g
        self.m = mass
        self.J = J.double()

    def get_control(self, parameters, state, ref_state, feed_forward_quantity):
        device = state.device
        desired_position, desired_velocity, desired_acceleration, \
          desired_pose, desired_angular_vel, desired_angular_acc = ref_state

        # extract specific state from state tensor
        position = state[0:3]
        pose = state[3:7]
        vel = state[7:10]
        angular_vel = state[10:13]

        # extract parameters
        kp, kv, kori, kw = parameters

        pose = torch.t(torch.atleast_2d(pose))
        angular_vel = torch.t(torch.atleast_2d(angular_vel))
        desired_angular_vel = torch.t(torch.atleast_2d(desired_angular_vel))
        desired_angular_acc = torch.t(torch.atleast_2d(desired_angular_acc))

        Rwb = quaternion_2_rotation_matrix(pose)

        err_position = torch.t(torch.atleast_2d(position - desired_position))
        err_vel = torch.t(torch.atleast_2d(vel - desired_velocity))
        desired_acceleration = torch.t(torch.atleast_2d(desired_acceleration))

        # compute the desired thrust
        b3_des = - kp * err_position - kv * err_vel - self.m * self.g * self.e3 \
          + self.m * desired_acceleration
        f = -torch.mm(b3_des.T, torch.mm(Rwb, self.e3))

        # compute the desired torque
        err_pose = (torch.mm(desired_pose.T, Rwb) - torch.mm(Rwb.T, desired_pose))
        err_pose = 0.5 * vee(err_pose)
        err_angular_vel = angular_vel \
          - torch.mm(Rwb.T, torch.mm(desired_pose, desired_angular_vel))

        M = - kori * err_pose - kw * err_angular_vel \
          + torch.cross(angular_vel, torch.mm(self.J, angular_vel))
        temp_M = torch.mm(hat(angular_vel),
                           torch.mm(Rwb.T, torch.mm(desired_pose, desired_angular_vel))) \
                            - torch.mm(Rwb.T, torch.mm(desired_pose, desired_angular_acc))
        M = M - torch.mm(self.J, temp_M)

        return torch.concat(
          [torch.max(torch.tensor([0.], device=device), f[0]), M[0], M[1], M[2]]
        )
