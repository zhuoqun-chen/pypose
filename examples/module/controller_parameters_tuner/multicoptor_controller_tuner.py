import torch
import argparse, os
import pypose as pp
import matplotlib.pyplot as plt
from pypose.module.geometric_controller import GeometricController
from examples.module.dynamics.multicopter import MultiCopter
from pypose.module.controller_parameters_tuner import ControllerParametersTuner
from min_snap_trajectory import PolynomialTrajectoryGenerator, WayPoint

def get_ref_states(initial_state, waypoints, dt):
    device = initial_state.device

    traj_gen = PolynomialTrajectoryGenerator()
    desired_states = traj_gen.generate_trajectory(waypoints, 0.1, 7)

    pose = torch.atleast_2d(initial_state[3:7])

    # get ref states
    ref_states = []
    last_ref_pose = pp.LieTensor(pose, ltype=pp.SO3_type).matrix()[0]
    last_ref_angle_dot = torch.zeros(3, device=device).double()
    last_ref_angle_ddot = torch.zeros(3, device=device).double()
    last_velocity_tensor = torch.zeros(3, device=device).double()

    gravity_acc_tensor = torch.stack([
        torch.tensor(0., device=device),
        torch.tensor(0., device=device),
        torch.tensor(g, device=device)]
    ).double()
    for index, waypoint in enumerate(desired_states[1:]):
        position_tensor = torch.concat(
            [torch.tensor(waypoint.position.x),
             torch.tensor(waypoint.position.y),
             torch.tensor(waypoint.position.z)]).double()

        # velocity computation
        velocity_tensor = torch.concat(
            [torch.tensor(waypoint.vel.x),
             torch.tensor(waypoint.vel.y),
             torch.tensor(waypoint.vel.z)]).double()

        # acceleration computation
        raw_acc_tensor = torch.concat(
            [torch.tensor(waypoint.acc.x),
             torch.tensor(waypoint.acc.y),
             torch.tensor(waypoint.acc.z)]).double()

        # minus gravity acc if choose upwards as the positive z-axis
        acc_tensor = raw_acc_tensor - gravity_acc_tensor
        acc_tensor_in_column_vector = torch.unsqueeze(acc_tensor, dim=1)

        # assume the yaw angle stays at 0
        b1_yaw_tensor = torch.stack([
            torch.tensor([1.], device=device),
            torch.tensor([0.], device=device),
            torch.tensor([0.], device=device)]).double()

        b3_ref = -acc_tensor_in_column_vector / torch.norm(acc_tensor_in_column_vector)
        b2_ref = torch.cross(b3_ref, b1_yaw_tensor)
        b2_ref = b2_ref / torch.norm(b2_ref)
        b1_ref = torch.cross(b2_ref, b3_ref)
        Rwb = (torch.concat([b1_ref, b2_ref, b3_ref], dim=1)).double()
        R_err = torch.mm(Rwb.T, torch.t(last_ref_pose.T))
        q_err = pp.from_matrix(R_err, ltype=pp.SO3_type)
        axis = torch.squeeze(torch.tensor([q_err[0], q_err[1], q_err[2]])).reshape([3])
        if torch.norm(axis) != 0:
            axis = axis / torch.norm(axis)
        angle = 2 * torch.acos(q_err[3])
        angle_dot = angle / dt * axis
        angle_ddot = ((angle_dot - last_ref_angle_dot) / dt).double()

        # assign zero pose to all waypoints
        # angle = 0.0
        # angle_dot = torch.zeros([3,], device=device).double()
        # angle_ddot = ((angle_dot - last_ref_angle_dot) / dt).double()

        ref_states.append((position_tensor, velocity_tensor,
                           raw_acc_tensor, Rwb, angle_dot, angle_ddot))

        last_ref_pose = Rwb
        last_ref_angle_dot = angle_dot
        last_ref_angle_ddot = angle_ddot
        last_velocity_tensor = velocity_tensor

    return ref_states

def run(dynamic_system, controller, controller_parameters, initial_state, ref_states, dt):
    system_states = []
    system_state = torch.clone(initial_state)
    system_states.append(system_state)
    for index, ref_state in enumerate(ref_states):
        controller_input = controller.get_control(controller_parameters, system_state, ref_state, None)
        system_new_state = dynamic_system.state_transition(system_state, controller_input, dt)

        system_state = system_new_state
        system_states.append(system_state)
    return system_states

def compute_loss(dynamic_system, controller, controller_parameters, penalty_coefficient,
                 initial_state, ref_states, dt):
    loss = 0
    system_state = torch.clone(initial_state)
    for index, ref_state in enumerate(ref_states):
      ref_position, ref_velocity, ref_acceleration, \
          ref_pose, ref_angular_vel, ref_angular_acc = ref_state
      controller_input = \
          controller.get_control(controller_parameters, system_state, ref_state, None)
      system_new_state = dynamic_system.state_transition(system_state,
                                                         controller_input, dt)

      position, pose, vel, angular_vel = system_new_state[0:3], system_new_state[3:7], \
          system_new_state[7:10], system_new_state[10:13]

      system_state = system_new_state

      loss += torch.norm(
        ref_position - position
      )

      loss += penalty_coefficient * torch.norm(controller_input)
    return loss / len(ref_states)


def func_to_get_state_error(state, ref_state):
    ref_position, ref_velocity, ref_acceleration, \
      ref_pose, ref_angular_vel, ref_angular_acc = ref_state

    ref_pose_SO3 = pp.from_matrix(ref_pose, ltype=pp.SO3_type).tensor()

    return state - torch.concat(
       [
          ref_position,
          ref_pose_SO3,
          ref_velocity,
          ref_angular_vel
       ]
    )

def get_sub_states(input_states, sub_state_index):
    sub_states = []
    for states in input_states:
        sub_states.append(states[sub_state_index].detach().cpu().item())
    return sub_states

def get_sub_ref_states(input_states, sub_state_index, tensor_item_index):
    sub_states = []
    for states in input_states:
        sub_states.append(states[sub_state_index][tensor_item_index].detach().cpu().item())
    return sub_states

def subPlot(ax, x, y, y_ref, xlabel=None, ylabel=None):
    res = []
    for i in range(0, len(y)):
        res.append(y_ref[i] - y[i])
    ax.plot(x, res, label=ylabel)
    ax.legend()
    ax.set_xlabel(xlabel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multicopter Controller Tuner Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/controller_parameters_tuner/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    g = 9.81

    # program parameters
    time_interval = 0.02
    learning_rate = 0.01

    initial_state = torch.zeros(13, device=args.device).double()
    initial_state[6] = 1
    # initial_controller_parameters = torch.ones(4, device=args.device).double()
    initial_controller_parameters = torch.tensor([6, 0.0100, 1.0000, 1.0000], device=args.device).double()

    points = torch.tensor([[[0., 0., 0.],
                            [1., 1., 0]]], device=args.device)
    waypoints = [
        WayPoint(0, 0, 0, 0),
        WayPoint(0, 0, -1, 1),
        WayPoint(0, 0, -3, 2)
    ]

    waypoints = pp.chspline(points, time_interval)[0]

    ref_states = get_ref_states(initial_state, waypoints, time_interval)

    e3 = torch.stack([
        torch.tensor([0.]),
        torch.tensor([0.]),
        torch.tensor([1.])]
      ).to(device=args.device).double()
    inertia = torch.tensor([
                            [0.0829, 0., 0.],
                            [0, 0.0845, 0],
                            [0, 0, 0.1377]
                          ], device=args.device)
    multicopter = MultiCopter(0.6, torch.tensor(g), inertia, e3)

    # start to tune the controller parameters
    penalty_coefficient = 0
    tuner = ControllerParametersTuner(learning_rate=learning_rate,
                                      penalty_coefficient=penalty_coefficient,
                                      device=args.device)

    controller = GeometricController(multicopter.m, multicopter.J, e3)
    controller_parameters = torch.clone(initial_controller_parameters)

    # only tune positions
    states_to_tune = torch.zeros([len(initial_state), len(initial_state)]
        , device=args.device)
    # only to tune the controller parameters dependending on the position error
    states_to_tune[0, 0] = 1
    states_to_tune[1, 1] = 1
    states_to_tune[2, 2] = 1

    last_loss_after_tuning = compute_loss(multicopter, controller, controller_parameters,
                                          penalty_coefficient, initial_state, ref_states, time_interval)
    print("Original Loss: ", last_loss_after_tuning)
    meet_termination_condition = False
    while not meet_termination_condition:
        controller_parameters = tuner.tune(
          multicopter,
          initial_state,
          ref_states,
          controller,
          controller_parameters,
          (0.01 * torch.ones_like(controller_parameters),
            10 * torch.ones_like(controller_parameters)),
          time_interval,
          states_to_tune,
          func_to_get_state_error
        )
        print("Controller parameters: ", controller_parameters)
        loss = compute_loss(multicopter, controller, controller_parameters, penalty_coefficient,
                                     initial_state, ref_states, time_interval)
        print("Loss: ", loss)

        if (last_loss_after_tuning - loss) < 0.0001:
            meet_termination_condition = True
            print("Meet tuning termination condition, terminated.")
        else:
            last_loss_after_tuning = loss

    # plot the result
    # get the result with original and tuned controller parameters
    original_system_states = run(multicopter, controller, torch.clone(initial_controller_parameters),
                                initial_state, ref_states, time_interval)
    new_system_states = run(multicopter, controller, controller_parameters, initial_state,
                            ref_states, time_interval)
    #insert intial reference state
    # get ref states
    initial_pose = torch.atleast_2d(torch.tensor([0, 0, 0, 1], device=args.device))
    # get ref states
    last_ref_pose = pp.LieTensor(initial_pose, ltype=pp.SO3_type).matrix()[0]
    last_ref_angle_dot = torch.zeros(3, device=args.device).double()
    last_ref_angle_ddot = torch.zeros(3, device=args.device).double()
    ref_states.insert(0,
      (torch.zeros(3, device=args.device).double(),
      torch.zeros(3, device=args.device).double(),
      torch.zeros(3, device=args.device).double(),
      last_ref_pose, last_ref_angle_dot, last_ref_angle_ddot)
    )

    time = torch.arange(0, len(waypoints) - 1 + time_interval, time_interval).numpy()
    f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].set_ylabel("Tracking error before tuning")
    ax[1].set_ylabel("Tracking error after tuning")
    subPlot(ax[0], time, get_sub_states(original_system_states, 0),  get_sub_ref_states(ref_states, 0, 0), ylabel='X')
    subPlot(ax[0], time, get_sub_states(original_system_states, 1),  get_sub_ref_states(ref_states, 0, 1), ylabel='Y')
    subPlot(ax[0], time, get_sub_states(original_system_states, 2),  get_sub_ref_states(ref_states, 0, 2), ylabel='Z')
    subPlot(ax[1], time, get_sub_states(new_system_states, 0), get_sub_ref_states(ref_states, 0, 0), ylabel='X')
    subPlot(ax[1], time, get_sub_states(new_system_states, 1), get_sub_ref_states(ref_states, 0, 1), ylabel='Y')
    subPlot(ax[1], time, get_sub_states(new_system_states, 2), get_sub_ref_states(ref_states, 0, 2), ylabel='Z')
    figure = os.path.join(args.save + 'multicoptor_controller_tuner.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
