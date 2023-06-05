import argparse, os
import torch
from examples.module.controller.SE3_controller import SE3Controller
from examples.module.dynamics.multicopter import MultiCopter
from pypose.optim.controller_parameters_tuner import ControllerParametersTuner
from examples.module.controller_parameters_tuner.waypoint import WayPoint
from examples.module.controller_parameters_tuner.trajectory_gen \
  import MinimumSnapTrajectoryGenerator
from examples.module.controller_parameters_tuner.commons \
  import quaternion_2_rotation_matrix, rotation_matrix_2_quaternion


def get_ref_states(initial_state, waypoints, dt):
    device = initial_state.device
    traj_generator = MinimumSnapTrajectoryGenerator()
    quad_trajectory = traj_generator.generate_trajectory(waypoints, dt, 7)

    # get ref states
    ref_states = []
    last_ref_pose = quaternion_2_rotation_matrix(initial_state[3:7])
    last_ref_angle_dot = torch.zeros([3, 1], device=device).double()
    last_ref_angle_ddot = torch.zeros([3, 1], device=device).double()
    ref_states.append(
      (torch.zeros([3, 1], device=device).double(),
      torch.zeros([3, 1], device=device).double(),
      torch.zeros([3, 1], device=device).double(),
      last_ref_pose, last_ref_angle_dot, last_ref_angle_ddot)
    )
    for wp in quad_trajectory[1:]:
        position_tensor = torch.stack(
            [torch.tensor([wp.position.x], device=device),
             torch.tensor([wp.position.y], device=device),
             torch.tensor([wp.position.z], device=device)]).double()
        velocity_tensor = torch.stack(
            [torch.tensor([wp.vel.x], device=device),
             torch.tensor([wp.vel.y], device=device),
             torch.tensor([wp.vel.z], device=device)]).double()
        # minus gravity acc if choose upwards as the positive z-axis
        raw_acc_tensor = torch.stack(
            [torch.tensor([wp.acc.x], device=device),
             torch.tensor([wp.acc.y], device=device),
             torch.tensor([wp.acc.z], device=device)]).double()
        acc_tensor = torch.stack([
            torch.tensor([wp.acc.x], device=device),
            torch.tensor([wp.acc.y], device=device),
            torch.tensor([wp.acc.z - g], device=device)]).double()

        # assume the yaw angle stays at 0
        b1_yaw_tensor = torch.stack([
            torch.tensor([1.], device=device),
            torch.tensor([0.], device=device),
            torch.tensor([0.], device=device)]).double()

        b3_ref = (-acc_tensor / torch.norm(acc_tensor)).double()
        b2_ref = torch.cross(b3_ref, b1_yaw_tensor)
        b2_ref = b2_ref / torch.norm(b2_ref)
        b1_ref = torch.cross(b2_ref, b3_ref)
        pose = (torch.concat([b1_ref, b2_ref, b3_ref], dim=1)).double()
        R_err = torch.mm(pose, torch.t(last_ref_pose))
        q_err = rotation_matrix_2_quaternion(R_err)
        q_err = q_err / torch.norm(q_err)
        axis = torch.stack([
            torch.tensor([q_err[1]], device=device),
            torch.tensor([q_err[2]], device=device),
            torch.tensor([q_err[3]], device=device)]).double()
        if torch.norm(axis) != 0:
          axis = axis / torch.norm(axis)
        # angle = 2 * torch.acos(q_err[0])
        # angle_dot = angle / dt * axis
        # angle_ddot = ((angle_dot - last_ref_angle_dot) / dt).double()
        angle = 0.0
        angle_dot = 0.0 * axis
        angle_ddot = ((angle_dot - last_ref_angle_dot) / dt).double()

        ref_states.append((position_tensor, velocity_tensor,
                           raw_acc_tensor, pose, angle_dot, angle_ddot))

        last_ref_pose = pose
        last_ref_angle_dot = angle_dot
        last_ref_angle_ddot = angle_ddot

    return ref_states


def compute_loss(dynamic_system, controller, controller_parameters,
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
    return loss / len(ref_states)


def func_to_get_state_error(state, ref_state):
    ref_position, ref_velocity, ref_acceleration, \
      ref_pose, ref_angular_vel, ref_angular_acc = ref_state

    return state - torch.vstack(
       [
          ref_position,
          rotation_matrix_2_quaternion(ref_pose),
          ref_velocity,
          ref_angular_vel
       ]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cartpole Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/dynamics/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    g = 9.81

    # program parameters
    time_interval = 0.02
    learning_rate = 0.1
    initial_state = torch.t(torch.tensor([[
        torch.tensor([0.]), # position.x
        torch.tensor([0.]), # position.y
        torch.tensor([0.]), # position.z
        torch.tensor([1.]), # quaternion.w
        torch.tensor([0.]), # quaternion.x
        torch.tensor([0.]), # quaternion.y
        torch.tensor([0.]), # quaternion.z
        torch.tensor([0.]), # vel.x
        torch.tensor([0.]), # vel.y
        torch.tensor([0.]), # vel.z
        torch.tensor([0.]), # anguler_vel.x
        torch.tensor([0.]), # anguler_vel.y
        torch.tensor([0.])]] # anguler_vel.z
      , device=args.device).double())
    initial_controller_parameters = torch.t(torch.tensor([[
        torch.tensor([1.]),
        torch.tensor([1.]),
        torch.tensor([1.]),
        torch.tensor([1.])]]
      , device=args.device).double())

    quadrotor_waypoints = [
        WayPoint(0, 0, 0, 0),
        WayPoint(1, 2, -1, 2),
        WayPoint(3, 6, -5, 4)
    ]

    ref_states = get_ref_states(initial_state, quadrotor_waypoints, time_interval)

    e3 = torch.stack([
        torch.tensor([0.]),
        torch.tensor([0.]),
        torch.tensor([1.])]
      ).to(device=args.device)
    multicopter = MultiCopter(time_interval,
                               0.6,
                               torch.tensor(g),
                               torch.tensor([
                                  [0.0829, 0., 0.],
                                  [0, 0.0845, 0],
                                  [0, 0, 0.1377]
                                ], device=args.device),
                                e3)

    # start to tune the controller parameters
    max_tuning_iterations = 30
    tuning_times = 0
    tuner = ControllerParametersTuner(learning_rate=learning_rate, device=args.device)

    controller = SE3Controller(multicopter.m, multicopter.J, e3)
    controller_parameters = torch.clone(initial_controller_parameters)

    # only tune positions
    states_to_tune = torch.zeros([len(initial_state), len(initial_state)]
        , device=args.device)
    states_to_tune[0, 0] = 1
    states_to_tune[1, 1] = 1
    states_to_tune[2, 2] = 1

    print("Original Loss: ", compute_loss(multicopter, controller, controller_parameters,
                                     initial_state, ref_states, time_interval))

    while tuning_times < max_tuning_iterations:
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
        tuning_times += 1
        # print("Controller parameters: ", controller_parameters)
        print("Loss: ", compute_loss(multicopter, controller, controller_parameters,
                                     initial_state, ref_states, time_interval))
