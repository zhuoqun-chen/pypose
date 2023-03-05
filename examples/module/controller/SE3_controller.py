import torch
from pypose.module.controller import Controller

def hat(vector):
    vector = vector.reshape([3, 1])
    return torch.stack([
      torch.stack([torch.tensor([0.]), -vector[2], vector[1]], dim=0),
      torch.stack([vector[2], torch.tensor([0.]), -vector[0]], dim=0),
      torch.stack([-vector[1], vector[0], torch.tensor([0.])], dim=0)
    ]).reshape([3, 3])

def vee(skew_symmetric_matrix):
    return torch.stack(
        (-skew_symmetric_matrix[1, 2],
        skew_symmetric_matrix[0, 2],
        -skew_symmetric_matrix[0, 1])
    ).reshape([3, 1])

def quaternion_2_rotation_matrix(q):
    q = q / torch.norm(q)
    qahat = hat(q[1:4])
    return (torch.eye(3) + 2 * torch.mm(qahat, qahat) + 2 * q[0] * qahat).double()

def angular_speed_2_quaternion_dot(quaternion, angular_speed):
    p, q, r = angular_speed
    zero_t = torch.tensor([0.])
    return -0.5 * torch.mm(torch.stack(
      [
        torch.stack([zero_t, p, q, r]),
        torch.stack([-p, zero_t, -r, q]),
        torch.stack([-q, r, zero_t, -p]),
        torch.stack([-r, -q, p, zero_t])
      ]).reshape([4, 4]), quaternion)

class SE3Controller(Controller):
    def __init__(self, mass, J, g = 9.81, e3 = torch.tensor([0., 0., 1.]).reshape([3, 1])):
        super(SE3Controller, self).__init__()
        self.e3 = e3.double()
        self.g = g
        self.m = mass
        self.J = J.double()

    def get_control(self, parameters, state, ref_state, feed_forward_quantity):
        desired_position, desired_velocity, desired_acceleration, \
          desired_pose, desired_angular_vel, desired_angular_acc = ref_state
    
        # extract specific state from state tensor
        position = state[0:3]
        pose = state[3:7]
        vel = state[7:10]
        angular_vel = state[10:13]
        
        # extract parameters
        kp, kv, kori, kw = parameters

        Rwb = quaternion_2_rotation_matrix(pose)

        err_position = position - desired_position
        err_vel = vel - desired_velocity
        
        # compute the desired thrust
        b3_des = - kp * err_position - kv * err_vel - self.m * self.g * self.e3 + self.m * desired_acceleration
        f = -torch.mm(b3_des.T, torch.mm(Rwb, self.e3))

        # compute the desired torque
        err_pose = (torch.mm(desired_pose.T, Rwb) - torch.mm(Rwb.T, desired_pose))
        err_pose = 0.5 * vee(err_pose)
        err_angular_vel = angular_vel - torch.mm(Rwb.T, torch.mm(desired_pose, desired_angular_vel))
        M = - kori * err_pose - kw * err_angular_vel + torch.cross(angular_vel, torch.mm(self.J, angular_vel))
        temp_M = torch.mm(hat(angular_vel), torch.mm(Rwb.T, torch.mm(desired_pose, desired_angular_vel))) - torch.mm(Rwb.T, torch.mm(desired_pose, desired_angular_acc))
        M = M - torch.mm(self.J, temp_M)

        return torch.stack(
          [torch.max(torch.tensor([0.]), f[0]), M[0], M[1], M[2]]
        ).reshape([4, 1])
    
    