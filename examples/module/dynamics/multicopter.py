import torch
from pypose.module.dynamics import NLS
from examples.module.controller_parameters_tuner.commons \
    import hat, vee, angular_vel_2_quaternion_dot, quaternion_2_rotation_matrix


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
        k1 = self.derivative(state, input)
        k2 = self.derivative(self.euler_update(state, k1, t / 2), input)
        k3 = self.derivative(self.euler_update(state, k2, t / 2), input)
        k4 = self.derivative(self.euler_update(state, k3, t), input)

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

    def derivative(self, state, input):
        position, pose, vel, angular_speed = state[0:3], state[3:7], \
            state[7:10], state[10:13]
        thrust, M = input[0], input[1:4]

        # convert the 1d row vector to 2d column vector
        M = torch.t(torch.atleast_2d(M))
        pose = torch.t(torch.atleast_2d(pose))

        pose_in_R = quaternion_2_rotation_matrix(pose)

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
