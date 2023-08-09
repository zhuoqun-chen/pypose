import math
import pypose as pp
import torch as torch

from pypose.lietensor.basics import vec2skew


def test_dynamics_cartpole():
    """
    Manually generate a trajectory for a forced cart-pole system and
    compare the trajectory and linearization.
    The reference data is originally obtained from the cartpole case
    in the examples folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The reference data
    state_ref = torch.tensor([
        [0.000000000000000000e+00, 0.000000000000000000e+00, 3.141592653589793116e+00, 0.000000000000000000e+00],
        [0.000000000000000000e+00, 4.004595033211845673e-18, 3.141592653589793116e+00, 8.009190066423691346e-18],
        [4.004595033211845793e-20, 4.444370253226554788e-06, 3.141592653589793116e+00, 2.222185126625291286e-06],
        [4.444370253230559533e-08, 1.333266620835869948e-05, 3.141592675811644586e+00, 6.666333104197371212e-06],
        [1.777703646158925914e-07, 2.666327252216060868e-05, 3.141592742474975442e+00, 1.333054627928972545e-05]],
        device=device)
    A_ref = torch.tensor([[
        [1.0, 0.01, 0.0, 0.0],
        [0.0, 1.0, -0.03270001922006555, -3.4808966152769563e-07],
        [0.0, 0.0, 1.0, 0.01],
        [0.0, 0.0, -0.06539991042629863, 0.9999998259560131]]],
        device=device)
    B_ref = torch.tensor([[0.0], [0.00044444299419410527], [0.0], [0.00022222042025532573]], device=device)
    C_ref = torch.eye(4, device=device)
    D_ref = torch.zeros((4,1), device=device)
    c1_ref = torch.tensor([0.0, 0.10273013763290852, 0.0, 0.20545987669936888], device=device)
    c2_ref = torch.zeros((4,), device=device)

    # The class
    class CartPole(pp.module.NLS):
        def __init__(self):
            super().__init__()
            self._tau = 0.01
            self._length = 1.5
            self._cartmass = 20.0
            self._polemass = 10.0
            self._gravity = 9.81
            self._polemassLength = self._polemass * self._length
            self._totalMass = self._cartmass + self._polemass

        def state_transition(self, state, input, t=None):
            x, xDot, theta, thetaDot = state
            force = input.squeeze()
            costheta = torch.cos(theta)
            sintheta = torch.sin(theta)

            temp = (
                force + self._polemassLength * thetaDot**2 * sintheta
            ) / self._totalMass
            thetaAcc = (self._gravity * sintheta - costheta * temp) / (
                self._length * (4.0 / 3.0 - self._polemass * costheta**2 / self._totalMass)
            )
            xAcc = temp - self._polemassLength * thetaAcc * costheta / self._totalMass

            _dstate = torch.stack((xDot, xAcc, thetaDot, thetaAcc))

            return state + torch.mul(_dstate, self._tau)

        def observation(self, state, input, t=None):
            return state

    # Time and input
    dt = 0.01
    N  = 1000
    time  = torch.arange(0, N + 1, device=device) * dt
    input = torch.sin(time)
    # Initial state
    state = torch.tensor([0, 0, math.pi, 0], device=device)

    # Create dynamics solver object
    cartPoleSolver = CartPole()

    # Calculate trajectory
    state_all = torch.zeros(N + 1, 4, device=device)
    state_all[0, :] = state
    for i in range(N):
        state_all[i + 1], _ = cartPoleSolver.forward(state_all[i], input[i])

    assert torch.allclose(state_ref, state_all[:5], rtol=1e-2)

    # Jacobian computation - Find jacobians at the last step
    jacob_state, jacob_input = state_all[-1], input[-1]
    cartPoleSolver.set_refpoint(state=jacob_state, input=jacob_input.unsqueeze(0), t=time[-1])

    assert torch.allclose(A_ref, cartPoleSolver.A)
    assert torch.allclose(B_ref, cartPoleSolver.B)
    assert torch.allclose(C_ref, cartPoleSolver.C)
    assert torch.allclose(D_ref, cartPoleSolver.D)
    assert torch.allclose(c1_ref, cartPoleSolver.c1)
    assert torch.allclose(c2_ref, cartPoleSolver.c2)


def test_dynamics_floquet():
    """
    Manually generate a trajectory for a floquet system (which is time-varying)
    and compare the trajectory and linearization against alternative solutions.
    This is for testing a time-varying system.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N     = 80                    # Number of time steps
    idx   = 5                     # The step to compute jacobians
    time  = torch.arange(0, N + 1, device=device)  # Time steps
    state = torch.tensor([1, 1], device=device)  # Initial state

    # The reference data
    def f(x, t):
        cc = torch.cos(2 * math.pi * t / 100)
        ss = torch.sin(2 * math.pi *t / 100)
        ff = torch.atleast_1d(torch.sin(2 * math.pi * t / 50))
        A = torch.tensor([[   1., cc/10],
                        [cc/10,    1.]], device=device)
        B = torch.tensor([[ss],
                        [1.]], device=device)
        return A.matmul(x) + B.matmul(ff), A, B

    state_ref = torch.zeros(N + 1, 2, device=device)
    state_ref[0] = state
    for i in range(N):
        state_ref[i + 1], _, _ = f(state_ref[i], time[i])
    obser_ref = state_ref[:-1] + time[:-1].reshape(-1, 1)

    _, A0_N, B0_N = f(torch.tensor([0., 0.], device=device), torch.tensor(N, device=device))
    _, A0_i, B0_i = f(torch.tensor([0., 0.], device=device), torch.tensor(idx, device=device))
    c2_N = torch.ones(2, device=device) * N
    c2_i = torch.ones(2, device=device) * idx
    C0 = torch.eye(2, device=device)
    D0 = torch.zeros(2, 1, device=device)
    c1 = torch.zeros(2, device=device)

    # The class
    class Floquet(pp.module.NLS):
        def __init__(self):
            super(Floquet, self).__init__()

        def state_transition(self, state, input, t):
            cc = torch.cos(2*math.pi*t/100)
            ss = torch.sin(2*math.pi*t/100)
            A = torch.tensor([[   1., cc/10],
                            [cc/10,    1.]], device=device)
            B = torch.tensor([[ss],
                            [1.]], device=device)
            return state.matmul(A) + B.matmul(input)

        def observation(self, state, input, t):
            return state + t

    # Input
    input = torch.sin(2 * math.pi * time / 50)

    # Create dynamics solver object
    solver = Floquet().to(device)

    # Calculate trajectory
    state_all = torch.zeros(N + 1, 2, device=device)
    state_all[0] = state
    obser_all = torch.zeros(N, 2, device=device)

    for i in range(N):
        state_all[i + 1], obser_all[i] = solver(state_all[i], input[i])

    assert torch.allclose(state_all, state_ref, atol=1e-5)
    assert torch.allclose(obser_all, obser_ref)

    # # For debugging
    # import matplotlib.pyplot as plt
    # f, ax = plt.subplots(nrows=4, sharex=True)
    # for _i in range(2):
    #     ax[_i].plot(time, state_all[:,_i], label='pp')
    #     ax[_i].plot(time, state_ref[:,_i], label='np')
    #     ax[_i].set_ylabel(f'State {_i}')
    # for _i in range(2):
    #     ax[_i+2].plot(time[:-1], obser_all[:,_i], label='pp')
    #     ax[_i+2].plot(time[:-1], obser_ref[:,_i], label='np')
    #     ax[_i+2].set_ylabel(f'Observation {_i}')
    # ax[-1].set_xlabel('time')
    # ax[-1].legend()
    # plt.show()

    # Jacobian computation - at the last step
    # Note for c1, the values are supposed to be zero, but due to numerical
    # errors the values can be ~ 1e-7, and hence we increase the atol
    # Same story below
    solver.set_refpoint()
    assert torch.allclose(A0_N, solver.A)
    assert torch.allclose(B0_N, solver.B)
    assert torch.allclose(C0, solver.C)
    assert torch.allclose(D0, solver.D)
    assert torch.allclose(c1, solver.c1, atol=1e-6)
    assert torch.allclose(c2_N, solver.c2)

    # Jacobian computation - at the step idx
    solver.set_refpoint(state=state_all[idx], input=input[idx], t=time[idx])
    assert torch.allclose(A0_i, solver.A)
    assert torch.allclose(B0_i, solver.B)
    assert torch.allclose(C0, solver.C)
    assert torch.allclose(D0, solver.D)
    assert torch.allclose(c1, solver.c1, atol=1e-6)
    assert torch.allclose(c2_i, solver.c2)

def test_dynamics_multicopter():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    class MultiCopter(pp.module.NLS):
        def __init__(self, dt, mass, g, J, e3):
            super(MultiCopter, self).__init__()
            self.m = mass
            self.J = J.double()
            self.J_inverse = torch.inverse(self.J)
            self.g = g
            self.e3 = e3
            self.tau = dt

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
            pose_SO3 = pp.LieTensor(pose, ltype=pp.SO3_type)
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

    # generate inputs for dubincar system
    dt = 0.01
    N  = 5
    time = torch.arange(0, N + 1, device=device) * dt
    # it's hard to generate inputs by hands, instead we use SE3 controller to
    # get the inputs
    inputs = torch.tensor([
        [ 0.000, 0.000, 0.000, 0.1962],
        [ 1.9612e-01, -5.6856e-03,  2.8868e-03, -4.7894e-08],
        [ 1.9590e-01, -6.3764e-03,  3.2420e-03, -3.8333e-07],
        [ 1.9560e-01,  6.7747e-04, -3.3588e-04, -4.2895e-07],
        [ 1.9523e-01,  7.0193e-03, -3.5589e-03,  3.3822e-07]
    ], device=device).double()

    # Initial state
    state = torch.zeros(13, device=device).double()
    state[6] = 1

    # Create dynamics solver object
    e3 = torch.stack([
        torch.tensor([0.]),
        torch.tensor([0.]),
        torch.tensor([1.])]
      ).to(device=device).double()
    g = 9.81
    mass = 0.19
    multicopterSolver = MultiCopter(dt,
                               torch.tensor(g, device=device),
                               torch.tensor(mass, device=device),
                               torch.tensor([
                                  [0.0829, 0., 0.],
                                  [0, 0.0845, 0],
                                  [0, 0, 0.1377]
                                ], device=device),
                                e3)

    # Calculate trajectory
    state_all = torch.zeros(N + 1, 13, device=device).double()
    state_all[0, :] = state
    for i in range(N):
        new_state, _ = multicopterSolver.forward(state_all[i, :], inputs[i, :])
        state_all[i+1, :] = new_state

    state_ref = torch.tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00,  1.9000e-03,
          0.0000e+00,  0.0000e+00,  1.4248e-02],
        [ 0.0000e+00,  0.0000e+00,  1.9000e-05,  0.0000e+00,  0.0000e+00,
          7.1242e-05,  1.0000e+00,  0.0000e+00,  0.0000e+00,  3.6001e-03,
         -6.8584e-04,  3.4163e-04,  1.4248e-02],
        [ 0.0000e+00,  0.0000e+00,  5.5001e-05, -3.4293e-06,  1.7079e-06,
          1.4248e-04,  1.0000e+00,  0.0000e+00,  0.0000e+00,  5.3004e-03,
         -1.4550e-03,  7.2524e-04,  1.4248e-02],
        [ 0.0000e+00,  0.0000e+00,  1.0800e-04, -1.0705e-05,  5.3333e-06,
          2.1373e-04,  1.0000e+00, -6.8088e-10, -1.3676e-09,  7.0010e-03,
         -1.3734e-03,  6.8535e-04,  1.4248e-02],
        [-6.8088e-12, -1.3676e-11,  1.7801e-04, -1.7572e-05,  8.7594e-06,
          2.8497e-04,  1.0000e+00, -2.8028e-09, -5.6289e-09,  8.7020e-03,
         -5.2673e-04,  2.6406e-04,  1.4248e-02]], device=device).double()

    assert torch.allclose(state_ref, state_all, rtol=1e-4)

     # Jacobian computation - at the last step
    jacob_state, jacob_input = state_all[-1, :], inputs[-1, :]
    multicopterSolver.set_refpoint(state=jacob_state, input=jacob_input, t=time[-1])

    A_ref = torch.tensor([[ 1.0000e+00,  0.0000e+00,  0.0000e+00,  7.3843e-10, -8.2921e-06,
          2.4169e-10,  1.9970e-11,  1.0000e-02,  0.0000e+00,  0.0000e+00,
         -1.9558e-11, -5.1826e-08,  1.2472e-13],
        [ 0.0000e+00,  1.0000e+00,  0.0000e+00,  8.2921e-06,  7.3843e-10,
         -1.2079e-10,  3.9889e-11,  0.0000e+00,  1.0000e-02,  0.0000e+00,
          5.1826e-08, -1.9507e-11, -6.5141e-14],
        [ 0.0000e+00,  0.0000e+00,  1.0000e+00, -2.5829e-10,  1.2892e-10,
          4.7182e-18, -2.6591e-15,  0.0000e+00,  0.0000e+00,  1.0000e-02,
         -1.4066e-12,  7.0466e-13,  9.1057e-20],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00,  7.1242e-05,
          3.9397e-06, -4.4096e-07,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          5.0000e-03, -2.5678e-06,  1.2271e-07],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -7.1242e-05,  1.0000e+00,
          7.9527e-06,  2.1598e-07,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          2.5799e-06,  5.0000e-03,  2.4780e-07],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.9410e-06, -7.9520e-06,
          1.0000e+00,  1.1874e-05,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         -8.7124e-08, -1.7730e-07,  5.0000e-03],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -7.9515e-06,  3.9421e-06,
         -7.1242e-05,  1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         -1.1089e-07,  5.4838e-08, -3.2059e-06],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  3.5445e-08, -1.9901e-04,
          3.9147e-10,  1.4750e-10,  1.0000e+00,  0.0000e+00,  0.0000e+00,
         -1.0920e-09, -2.4876e-06, -8.0528e-11],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.9901e-04,  3.5445e-08,
         -2.2232e-10,  2.7567e-10,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          2.4876e-06, -1.0880e-09,  3.9621e-11],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -2.5922e-09,  1.3106e-09,
          3.7006e-16, -1.1180e-13,  0.0000e+00,  0.0000e+00,  1.0000e+00,
          5.7399e-13,  5.7349e-14,  6.7475e-18],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00, -9.1437e-05,  5.0599e-06],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          9.2404e-05,  1.0000e+00,  1.0313e-05],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          9.1577e-08, -1.8480e-07,  1.0000e+00]], device=device).double()
    B_ref = torch.tensor([[-6.6043e-10, -2.8949e-12, -6.1332e-09, -1.0830e-13],
        [-1.3229e-09,  6.2516e-09, -2.8401e-12,  5.3588e-14],
        [-4.2474e-05,  2.8879e-14, -1.3365e-14,  1.1830e-20],
        [ 0.0000e+00,  1.5078e-03, -8.2263e-07,  1.7597e-08],
        [ 0.0000e+00,  8.4094e-07,  1.4793e-03,  3.5543e-08],
        [ 0.0000e+00, -1.6325e-08, -3.2929e-08,  9.0777e-04],
        [ 0.0000e+00, -3.3445e-08,  1.6213e-08, -5.8204e-07],
        [-6.7130e-09, -2.4482e-10, -4.9066e-07, -1.7543e-11],
        [-1.3278e-08,  5.0013e-07, -2.3960e-10,  8.6548e-12],
        [-1.0194e-03,  1.2255e-11, -5.8989e-12,  1.0009e-18],
        [ 0.0000e+00,  1.2063e-01, -2.7052e-05,  1.3274e-06],
        [ 0.0000e+00,  2.7866e-05,  1.1834e-01,  2.7033e-06],
        [ 0.0000e+00,  3.9915e-08, -7.8931e-08,  7.2622e-02]], device=device).double()
    C_ref = torch.eye(13, device=device).double()
    D_ref = torch.zeros((13,4), device=device).double()
    c1_ref = torch.tensor([ 1.6402e-11,  3.2768e-11,  7.9167e-05,  7.6009e-09,  4.3808e-07,
        -2.2178e-07, -1.1874e-05,  4.3814e-10,  8.7343e-10,  1.9000e-03,
        -7.2095e-08, -1.4695e-07, -2.3197e-10], device=device).double()
    c2_ref = torch.zeros((13,), device=device).double()

    assert torch.allclose(A_ref, multicopterSolver.A, atol=1e-4)
    assert torch.allclose(B_ref, multicopterSolver.B, atol=1e-4)
    assert torch.allclose(C_ref, multicopterSolver.C)
    assert torch.allclose(D_ref, multicopterSolver.D)
    assert torch.allclose(c1_ref, multicopterSolver.c1, atol=1e-4)
    assert torch.allclose(c2_ref, multicopterSolver.c2)


def test_dynamics_dubincar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class DubinCar(pp.module.NLS):
        def __init__(self, dt):
            super(DubinCar, self).__init__()
            self.tau = dt

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

    state_ref = torch.tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [5.0000e-01, 1.2500e-03, 5.0000e-03, 1.0000e+00, 1.0000e-02],
        [1.9999e+00, 1.9999e-02, 2.0000e-02, 2.0000e+00, 2.0000e-02],
        [4.4985e+00, 1.0123e-01, 4.5000e-02, 3.0000e+00, 3.0000e-02],
        [7.9915e+00, 3.1983e-01, 8.0000e-02, 4.0000e+00, 4.0000e-02],
        [1.2467e+01, 7.8023e-01, 1.2500e-01, 5.0000e+00, 5.0000e-02],
        [1.7903e+01, 1.6156e+00, 1.8000e-01, 6.0000e+00, 6.0000e-02],
        [2.4256e+01, 2.9863e+00, 2.4500e-01, 7.0000e+00, 7.0000e-02],
        [3.1457e+01, 5.0764e+00, 3.2000e-01, 8.0000e+00, 8.0000e-02],
        [3.9402e+01, 8.0897e+00, 4.0500e-01, 9.0000e+00, 9.0000e-02],
        [4.7943e+01, 1.2242e+01, 5.0000e-01, 1.0000e+01, 1.0000e-01]], device=device).double()

    # generate inputs for dubincar system
    N  = 10
    time = torch.squeeze(torch.ones((1, N), device=device))
    input = torch.stack([time, 0.01 * torch.ones_like(time)], dim=0).double()

    # Initial state
    state = torch.tensor([0, 0, 0, 0, 0], device=device).double()

    # Create dynamics solver object
    dubinCarSolver = DubinCar(1)

    # Calculate trajectory
    state_all = torch.zeros(N + 1, 5, device=device).double()
    state_all[0, :] = state
    for i in range(N):
        state_all[i + 1], _ = dubinCarSolver.forward(state_all[i], input[:, i])

    assert torch.allclose(state_ref, state_all, rtol=1e-2)

    # Jacobian computation - at the last step
    jacob_state, jacob_input = state_all[-1], input[:, -1]
    dubinCarSolver.set_refpoint(state=jacob_state, input=jacob_input, t=time[-1])

    A_ref = torch.tensor([[ 1.0000,  0.0000, -5.5080,  0.8513, -2.8759],
        [ 0.0000,  1.0000,  8.9336,  0.5239,  4.4895],
        [ 0.0000,  0.0000,  1.0000,  0.0000,  1.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000]],
        device=device).double()
    B_ref = torch.tensor([[ 0.4210, -0.9806],
        [ 0.2694,  1.4988],
        [ 0.0000,  0.5000],
        [ 1.0000,  0.0000],
        [ 0.0000,  1.0000]], device=device).double()
    C_ref = torch.eye(5, device=device).double()
    D_ref = torch.zeros((5, 2), device=device).double()
    c1_ref = torch.tensor([ 3.0514e+00, -4.9308e+00,  0.0000e+00,
                           1.1102e-16,  1.7347e-18], device=device).double()
    c2_ref = torch.zeros((5), device=device).double()

    assert torch.allclose(A_ref, dubinCarSolver.A, atol=1e-4)
    assert torch.allclose(B_ref, dubinCarSolver.B, atol=1e-4)
    assert torch.allclose(C_ref, dubinCarSolver.C)
    assert torch.allclose(D_ref, dubinCarSolver.D)
    assert torch.allclose(c1_ref, dubinCarSolver.c1, atol=1e-1)
    assert torch.allclose(c2_ref, dubinCarSolver.c2)


def test_dynamics_lti():

    """
    For a System with p inputs, q outputs, and n state variables,
    A, B, C, D are n*n n*p q*n and q*p constant matrices.
    N: channels

    A = torch.randn(N, n, n)
    B = torch.randn(N, n, p)
    C = torch.randn(N, q, n)
    D = torch.randn(N, q, p)
    c1 = torch.randn(N, 1, n)
    c2 = torch.randn(N, 1, q)
    state = torch.randn(N, 1, n)
    input = torch.randn(N, 1, p)
    """

    # The most general case that all parameters are in the batch.
    # The user could change the corresponding values according to the actual physical system and directions above.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A_1 = torch.randn(5, 4, 4, device=device)
    B_1 = torch.randn(5, 4, 2, device=device)
    C_1 = torch.randn(5, 3, 4, device=device)
    D_1 = torch.randn(5, 3, 2, device=device)
    c1_1 = torch.randn(5, 4, device=device)
    c2_1 = torch.randn(5, 3, device=device)
    state_1 = torch.randn(5, 4, device=device)
    input_1 = torch.randn(5, 2, device=device)

    lti_1 = pp.module.LTI(A_1, B_1, C_1, D_1, c1_1, c2_1).to(device)

    # The user can implement this line to print each parameter for comparison.

    z_1, y_1 = lti_1(state_1,input_1)

    z_1_ref = pp.bmv(A_1, state_1) + pp.bmv(B_1, input_1) + c1_1

    y_1_ref = pp.bmv(C_1, state_1) + pp.bmv(D_1, input_1) + c2_1

    assert torch.allclose(z_1, z_1_ref)
    assert torch.allclose(y_1, y_1_ref)


    #In this example, A, B, C, D, c1, c2 are single inputs, state and input are in a batch.

    A_2 = torch.randn(4, 4, device=device)
    B_2 = torch.randn(4, 2, device=device)
    C_2 = torch.randn(3, 4, device=device)
    D_2 = torch.randn(3, 2, device=device)
    c1_2 = torch.randn(4, device=device)
    c2_2 = torch.randn(3, device=device)
    state_2 = torch.randn(5, 4, device=device)
    input_2 = torch.randn(5, 2, device=device)

    lti_2 = pp.module.LTI(A_2, B_2, C_2, D_2, c1_2, c2_2).to(device)

    z_2, y_2 = lti_2(state_2, input_2)

    z_2_ref = pp.bmv(A_2, state_2) + pp.bmv(B_2, input_2) + c1_2
    y_2_ref = pp.bmv(C_2, state_2) + pp.bmv(D_2, input_2) + c2_2

    assert torch.allclose(z_2, z_2_ref)
    assert torch.allclose(y_2, y_2_ref)


    # In this example, all parameters are single inputs.

    A_3 = torch.randn(4, 4, device=device)
    B_3 = torch.randn(4, 2, device=device)
    C_3 = torch.randn(3, 4, device=device)
    D_3 = torch.randn(3, 2, device=device)
    c1_3 = torch.randn(4, device=device)
    c2_3 = torch.randn(3, device=device)
    state_3 = torch.randn(4, device=device)
    input_3 = torch.randn(2, device=device)

    lti_3 = pp.module.LTI(A_3, B_3, C_3, D_3, c1_3, c2_3).to(device)

    z_3, y_3 = lti_3(state_3, input_3)

    z_3_ref = A_3.mv(state_3) + B_3.mv(input_3) + c1_3
    y_3_ref = C_3.mv(state_3) + D_3.mv(input_3) + c2_3

    assert torch.allclose(z_3, z_3_ref)
    assert torch.allclose(y_3, y_3_ref)


if __name__ == '__main__':
    test_dynamics_cartpole()
    test_dynamics_floquet()
    test_dynamics_lti()
    test_dynamics_dubincar()
    test_dynamics_multicopter()
