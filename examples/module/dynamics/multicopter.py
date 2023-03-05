import torch
from pypose.module.dynamics import NLS


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

class MultiCopter(NLS):
    def __init__(self, mass, g, J):
        self.m = mass
        self.J = J.double()
        self.J_inverse = torch.inverse(self.J)
        self.g = g
        self.e3 = torch.tensor([0., 0., 1.]).reshape([3, 1])

    def state_transition(self, state, input, t=None):
        k1 = self.derivative(state, input)
        k2 = self.derivative(self.euler_update(state, k1, t / 2), input)
        k3 = self.derivative(self.euler_update(state, k2, t / 2), input)
        k4 = self.derivative(self.euler_update(state, k3, t), input)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * t
    
    def observation(self, state, input, t=None):
        return state
    
    def euler_update(self, state, derivative, dt):
        position, pose, vel, angular_speed = state[0:3], state[3:7], state[7:10], state[10:13]
        vel, angular_derivative, acceleration, w_dot = derivative[0:3], derivative[3:7], derivative[7:10], derivative[10:13]

        position_updated = position + vel * dt
        pose_updated = pose + angular_derivative * dt
        pose_updated = pose_updated / torch.norm(pose_updated)
        vel_updated = vel + acceleration * dt
        angular_speed_updated = angular_speed + w_dot * dt

        return torch.concat(
            [
                position_updated,
                pose_updated,
                vel_updated,
                angular_speed_updated
            ]
        )

    def derivative(self, state, input):
        position, pose, vel, angular_speed = state[0:3], state[3:7], state[7:10], state[10:13]
        thrust, M = input[0], input[1:4].reshape([3, 1])

        pose_in_R = self.quaternion_2_rotation_matrix(pose)

        acceleration = (torch.mm(pose_in_R, -thrust * self.e3) + self.m * self.g * self.e3) / self.m
        w_dot = torch.mm(self.J_inverse, M - torch.cross(angular_speed, torch.mm(self.J, angular_speed)))
        
        return torch.vstack([
                vel,
                angular_speed_2_quaternion_dot(pose, angular_speed),
                acceleration,
                w_dot
            ]
        )

    def quaternion_2_rotation_matrix(self, q):
        q = q / torch.norm(q)
        qahat = hat(q[1:4])
        return (torch.eye(3) + 2 * torch.mm(qahat, qahat) + 2 * q[0] * qahat).double()
  
    