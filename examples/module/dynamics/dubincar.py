import torch
from pypose.module.dynamics import NLS

class DubinCar(NLS):
    def __init__(self):
        super(DubinCar, self).__init__()

    # Use RK4 to infer the k+1 state
    def state_transition(self, state, input, t=None):
        return self.rk4(state, input, t)

    def rk4(self, state, input, t=None):
        k1 = self.xdot(state, input)
        k1_state = self.euler_update(state, k1, t / 2)

        k2 = self.xdot(k1_state, input)
        k2_state = self.euler_update(state, k2, t / 2)

        k3 = self.xdot(k2_state, input)
        k3_state = self.euler_update(state, k3, t)

        k4 = self.xdot(k3_state, input)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * t

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
