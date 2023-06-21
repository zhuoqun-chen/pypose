import torch
from pypose.module.dynamics import NLS

class DubinCar(NLS):
    def __init__(self, dt):
        super(DubinCar, self).__init__()
        self.tau = dt

    # Use RK4 to infer the k+1 state
    def state_transition(self, state, input, t=None):
        k1 = self.xdot(state, input)
        k2 = self.xdot(self.euler_update(state, k1, self.tau / 2), input)
        k3 = self.xdot(self.euler_update(state, k2, self.tau / 2), input)
        k4 = self.xdot(self.euler_update(state, k3, self.tau), input)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * self.tau

    def observation(self, state, input, t=None):
        return state

    def xdot(self, state, input):
        pos_x, pos_y, orientation, vel, w = state
        # acceleration and angular acceleration
        acc, w_dot = input

        return torch.stack(
            [
                vel * torch.cos(orientation),
                vel * torch.sin(orientation),
                w,
                acc,
                w_dot
            ]
        )

    def euler_update(self, state, derivative, dt):
        pos_x, pos_y, orientation, vel, w = state
        vx, vy, angular_speed, acc, w_dot = derivative
        return torch.stack(
          [
            pos_x + vx * dt,
            pos_y + vy * dt,
            orientation + angular_speed * dt,
            vel + acc * dt,
            w + w_dot * dt
          ]
        )
