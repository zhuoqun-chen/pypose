import torch
import pypose as pp
from pypose.module.dynamics import NLS
from pypose.lietensor import LieTensor

def angular_vel_2_quaternion_dot(quaternion, w):
    device = quaternion.device
    p, q, r = w
    zero_t = torch.tensor([0.], device=device)
    return -0.5 * torch.mm(torch.squeeze(torch.stack(
        [
            torch.stack([zero_t, -r, q, -p], dim=-1),
            torch.stack([r, zero_t, -p, -q], dim=-1),
            torch.stack([-q, p, zero_t, -r], dim=-1),
            torch.stack([p, q, r, zero_t], dim=-1)
        ])), quaternion)

class MultiCopter(NLS):
    def __init__(self, dt, mass, g, J, e3):
        super(MultiCopter, self).__init__()
        self.m = mass
        self.J = J.double()
        self.J_inverse = torch.inverse(self.J)
        self.g = g
        self.e3 = e3
        self.tau = dt

    def state_transition(self, state, input, t=None):
        k1 = self.xdot(state, input)
        k2 = self.xdot(self.euler_update(state, k1, t / 2), input)
        k3 = self.xdot(self.euler_update(state, k2, t / 2), input)
        k4 = self.xdot(self.euler_update(state, k3, t), input)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * self.tau

    def observation(self, state, input, t=None):
        return state

    def euler_update(self, state, derivative, dt):
        position, pose, vel, angular_speed = state[0:3], state[3:7], \
            state[7:10], state[10:13]
        vel, angular_derivative, acceleration, w_dot = derivative[0:3], derivative[3:7], \
            derivative[7:10], derivative[10:13]

        position_updated = position + vel * dt
        pose_updated = pose + angular_derivative * dt
        pose_updated = pose_updated / torch.norm(pose_updated)
        vel_updated = vel + acceleration * dt
        angular_speed_updated = angular_speed + w_dot * dt

        return torch.concat([
                position_updated,
                pose_updated,
                vel_updated,
                angular_speed_updated
            ]
        )

    def xdot(self, state, input):
        position, pose, vel, angular_speed = state[0:3], state[3:7], \
            state[7:10], state[10:13]
        thrust, M = input[0], input[1:4]

        # convert the 1d row vector to 2d column vector
        M = torch.t(torch.atleast_2d(M))
        pose = torch.atleast_2d(pose)
        pose_SO3 = LieTensor(pose, ltype=pp.SO3_type)
        pose_in_R = pose_SO3.matrix()[0]
        pose = torch.t(pose)

        acceleration = (torch.mm(pose_in_R, -thrust * self.e3)
                        + self.m * self.g * self.e3) / self.m

        angular_speed = torch.t(torch.atleast_2d(angular_speed))
        w_dot = torch.mm(self.J_inverse,
                        M - torch.cross(angular_speed, torch.mm(self.J, angular_speed)))

        return torch.concat([
                vel,
                torch.squeeze(torch.t(angular_vel_2_quaternion_dot(pose, angular_speed))),
                torch.squeeze(torch.t(acceleration)),
                torch.squeeze(torch.t(w_dot))
            ]
        )
