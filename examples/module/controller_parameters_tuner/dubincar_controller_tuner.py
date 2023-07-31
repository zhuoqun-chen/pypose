import torch
import argparse, os
import pypose as pp
import matplotlib.pyplot as plt
from pypose.module.dubincar_controller import DubinCarController
from examples.module.dynamics.dubincar import DubinCar
from pypose.module.controller_parameters_tuner import ControllerParametersTuner


def get_shortest_path_between_angles(original_ori, des_ori):
  e_ori = des_ori - original_ori
  if abs(e_ori) > torch.pi:
    if des_ori > original_ori:
      e_ori = - (original_ori + 2 * torch.pi - des_ori)
    else:
      e_ori = des_ori + 2 * torch.pi - original_ori
  return e_ori


def get_ref_states(waypoints, dt, device):
    car_desired_states = []
    last_desired_state = torch.zeros(9, device=device)

    for index, waypoint in enumerate(waypoints[1:]):
        # get path attributes from the last desired state
        last_waypoint = waypoints[index]
        vel = (waypoint - last_waypoint) / dt
        last_vel = torch.tensor([last_desired_state[2], last_desired_state[3]])
        acc = (vel[0:2] - last_vel) / dt
        last_desired_orientation, \
        last_desired_orientation_dot, \
        last_desired_orientation_ddot = last_desired_state[-3:]
        current_orientation = torch.atan2(vel[1], vel[0])
        last_desired_orientation_remaindar = \
          torch.atan2(torch.sin(last_desired_orientation), torch.cos(last_desired_orientation))
        delta_angle = get_shortest_path_between_angles(last_desired_orientation_remaindar, current_orientation)
        current_orientation = last_desired_orientation + delta_angle
        current_ori_dot = (current_orientation - last_desired_orientation) / dt

        car_desired_states.append(torch.tensor(
          [
            waypoint[0], \
            waypoint[1], \
            vel[0], \
            vel[1], \
            acc[0], \
            acc[1], \
            current_orientation, \
            current_ori_dot, \
            (current_ori_dot - last_desired_orientation_dot) / dt], device=device))
        last_desired_state = car_desired_states[-1:][0]
    return car_desired_states

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
        ref_position, ref_velocity, ref_acceleration, ref_pose, ref_angular_vel, ref_angular_acc = \
          ref_state[0:2], ref_state[2:4], ref_state[4:6], \
          ref_state[6:7], ref_state[7:8], ref_state[8:9]
        controller_input = controller.get_control(controller_parameters, system_state, ref_state, None)
        system_new_state = dynamic_system.state_transition(system_state, controller_input, dt)

        position_x, position_y, pose, vel, angular_vel = system_new_state

        system_state = system_new_state

        loss += torch.norm(
          ref_position - torch.stack([position_x, position_y])
        )
        loss += penalty_coefficient * torch.norm(controller_input)
    return loss / len(ref_states)

def func_to_get_state_error(state, ref_state):
    ref_position, ref_velocity, ref_acceleration, ref_pose, ref_angular_vel, ref_angular_acc = \
        ref_state[0:2], ref_state[2:4], ref_state[4:6], \
        ref_state[6], ref_state[7], ref_state[8]

    return state - torch.stack(
       [
          ref_position[0],
          ref_position[1],
          ref_pose,
          torch.norm(ref_velocity[0]),
          ref_angular_vel
       ])

def get_sub_states(input_states, sub_state_index):
    sub_states = []
    for states in input_states:
        sub_states.append(states[sub_state_index].detach().cpu().item())
    return sub_states

def subPlot(ax, x, y, y_ref, xlabel=None, ylabel=None):
    ax.plot(x, y, label='result')
    ax.plot(x, y_ref, label='reference')
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dubincar Controller Tuner Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/controller_parameters_tuner/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    # program parameters
    time_interval = 0.02
    learning_rate = 0.005
    # states tensor: x position, y position, orientation, velocity, angular_velocity
    initial_state = torch.zeros(5, device=args.device).double()
    # controller parameters: kp_position, kp_velocity, kp_orientation, kp_angular_velocity
    initial_controller_parameters = torch.ones(4, device=args.device).double()

    points = torch.tensor([[[0., 0., 0.],
                            [1., 0, 0],
                            [1., 1, 0],
                            [1., 2, 0],
                            [2., 3, 0],
                            [4., 3, 0]]])

    waypoints = pp.CSplineR3(points, time_interval)[0]
    ref_states = get_ref_states(waypoints, time_interval, args.device)

    dubincar = DubinCar(time_interval)

    # start to tune the controller parameters
    penalty_coefficient = 0.0001
    tuner = ControllerParametersTuner(learning_rate=learning_rate,
                                      penalty_coefficient=penalty_coefficient,
                                      device=args.device)

    controller = DubinCarController()
    controller_parameters = torch.clone(initial_controller_parameters)

    states_to_tune = torch.zeros([len(initial_state), len(initial_state)]
      , device=args.device)
    # only to tune the controller parameters dependending on the position error
    states_to_tune[0, 0] = 1
    states_to_tune[1, 1] = 1

    last_loss_after_tuning = compute_loss(dubincar, controller, controller_parameters,
                                          penalty_coefficient, initial_state, ref_states, time_interval)
    print("Original Loss: ", last_loss_after_tuning)

    meet_termination_condition = False
    while not meet_termination_condition:
        controller_parameters = tuner.tune(
          dubincar,
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

        loss = compute_loss(dubincar, controller, controller_parameters, penalty_coefficient,
                            initial_state, ref_states, time_interval)
        print("Loss: ", loss)

        if (last_loss_after_tuning - loss) < 0.0001:
            meet_termination_condition = True
            print("Meet tuning termination condition, terminated.")
        else:
            last_loss_after_tuning = loss

    # plot the result
    # get the result with original and tuned controller parameters
    original_system_states = run(dubincar, controller, torch.clone(initial_controller_parameters)
                                  , initial_state, ref_states, time_interval)
    new_system_states = run(dubincar, controller, controller_parameters, initial_state,
        ref_states, time_interval)
    ref_states.insert(0, torch.zeros(9, device=args.device))
    time = torch.arange(0, len(points[0]) - 1 + time_interval, time_interval).numpy()
    f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    subPlot(ax[0], time, get_sub_states(original_system_states, 0), get_sub_states(ref_states, 0), ylabel='X position')
    subPlot(ax[0], time, get_sub_states(original_system_states, 1), get_sub_states(ref_states, 1), ylabel='Y position')
    subPlot(ax[1], time, get_sub_states(new_system_states, 0), get_sub_states(ref_states, 0), ylabel='X after tuning')
    subPlot(ax[1], time, get_sub_states(new_system_states, 1), get_sub_states(ref_states, 1), ylabel='Y after tuning')
    figure = os.path.join(args.save + 'dubincar_controller_tuner.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
