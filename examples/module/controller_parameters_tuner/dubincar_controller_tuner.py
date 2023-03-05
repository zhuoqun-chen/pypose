import torch
from examples.module.controller_parameters_tuner.waypoint import WayPoint
from trajectory_gen import PolynomialTrajectoryGenerator

from examples.module.controller.dubincar_controller import DubinCarController
from examples.module.dynamics.dubincar import DubinCar
from pypose.optim.controller_parameters_tuner import ControllerParametersTuner

def get_shortest_path_between_angles(original_ori, des_ori):
  e_ori = des_ori - original_ori
  if abs(e_ori) > torch.pi:
    if des_ori > original_ori:
      e_ori = - (original_ori + 2 * torch.pi - des_ori)
    else:
      e_ori = des_ori + 2 * torch.pi - original_ori
  return e_ori

def get_ref_states(waypoints, dt):
    traj_gen = PolynomialTrajectoryGenerator()
    generated_waypoints = traj_gen.generate_trajectory(waypoints, time_interval, 7)

    car_desired_states = []
    last_desired_state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    for wp in generated_waypoints[1:]:
        last_desired_orientation, \
        last_desired_orientation_dot, \
        last_desired_orientation_ddot = last_desired_state[-3:]
        current_orientation = torch.atan2(torch.tensor(wp.vel.y), torch.tensor(wp.vel.x)).item()
        last_desired_orientation_remaindar = torch.atan2(torch.sin(torch.tensor(last_desired_orientation)), torch.cos(torch.tensor(last_desired_orientation))).item()
        delta_angle = get_shortest_path_between_angles(last_desired_orientation_remaindar, current_orientation)
        current_orientation = last_desired_orientation + delta_angle
        current_ori_dot = (current_orientation - last_desired_orientation) / dt
        
        car_desired_states.append(
          (
            wp.position.x, \
            wp.position.y, \
            wp.vel.x, \
            wp.vel.y, \
            wp.acc.x, \
            wp.acc.y, \
            current_orientation, \
            current_ori_dot, \
            (current_ori_dot - last_desired_orientation_dot) / dt))
        last_desired_state = car_desired_states[-1:][0]
    return car_desired_states

def compute_loss(dynamic_system, controller, controller_parameters, initial_state, ref_states, dt):
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
        torch.tensor(ref_position) - torch.tensor([position_x, position_y])
      )
    return loss / len(ref_states)

def func_to_get_state_error(state, ref_state):
    ref_position, ref_velocity, ref_acceleration, ref_pose, ref_angular_vel, ref_angular_acc = \
        ref_state[0:2], ref_state[2:4], ref_state[4:6], \
        ref_state[6], ref_state[7], ref_state[8]

    return state - torch.tensor(
       [
          ref_position[0],
          ref_position[1],
          ref_pose,
          torch.norm(torch.tensor(ref_velocity[0])),
          ref_angular_vel
       ]
    ).reshape((5, 1))

if __name__ == "__main__":
    # program parameters
    time_interval = 0.05
    learning_rate = 0.5
    initial_state = torch.tensor([0, 0, 0, 0, 0]).reshape([5, 1]).double()
    initial_controller_parameters = torch.tensor([10., 1., 1., 1.]).reshape([4, 1]).double()

    dubin_car_waypoints = [
      WayPoint(0, 0, 0, 0),
      WayPoint(1, 1, 0, 2),
      WayPoint(2, 0, 0, 4),
      WayPoint(3, -1, 0, 6),
      WayPoint(4, 0, 0, 8),
    ]

    ref_states = get_ref_states(dubin_car_waypoints, time_interval)

    dubincar = DubinCar(time_interval)
    
    # start to tune the controller parameters
    max_tuning_iterations = 30
    tuning_times = 0
    tuner = ControllerParametersTuner(learning_rate=learning_rate)
    
    controller = DubinCarController()
    controller_parameters = torch.clone(initial_controller_parameters)

    # only tune positions
    states_to_tune = torch.zeros([len(initial_state), len(initial_state)])
    states_to_tune[0, 0] = 1
    states_to_tune[1, 1] = 1
    
    while tuning_times < max_tuning_iterations:
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
        tuning_times += 1
        print("Controller parameters: ", controller_parameters)
        print("Loss: ", compute_loss(dubincar, controller, controller_parameters, initial_state, ref_states, time_interval))