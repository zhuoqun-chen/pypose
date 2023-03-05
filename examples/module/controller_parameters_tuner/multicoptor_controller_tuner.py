import torch
from examples.module.controller_parameters_tuner.waypoint import WayPoint
from trajectory_gen import PolynomialTrajectoryGenerator
from commons import quaternion_2_rotation_matrix, rotation_matrix_2_quaternion

from examples.module.controller.SE3_controller import SE3Controller
from examples.module.dynamics.multicopter import MultiCopter
from pypose.optim.controller_parameters_tuner import ControllerParametersTuner


def get_ref_states(waypoints, dt):
    traj_generator = PolynomialTrajectoryGenerator()
    quad_trajectory = traj_generator.generate_trajectory(waypoints, dt, 7)

    # get ref states
    ref_states = []
    last_ref_pose = quaternion_2_rotation_matrix(initial_state[3:7])
    last_ref_angle_dot = torch.zeros([3, 1]).double()
    last_ref_angle_ddot = torch.zeros([3, 1]).double()
    ref_states.append(
      (torch.zeros([3, 1]).double(), torch.zeros([3, 1]).double(), torch.zeros([3, 1]).double(), last_ref_pose, last_ref_angle_dot, last_ref_angle_ddot)
    )
    for wp in quad_trajectory[1:]:
        position_tensor = torch.tensor([wp.position.x, wp.position.y, wp.position.z]).reshape([3,1]).double()
        velocity_tensor = torch.tensor([wp.vel.x, wp.vel.y, wp.vel.z]).reshape([3,1]).double()
        # minus gravity acc if choose upwards as the positive z-axis 
        raw_acc_tensor = torch.tensor([wp.acc.x, wp.acc.y, wp.acc.z]).reshape([3,1]).double()
        acc_tensor = torch.tensor([wp.acc.x, wp.acc.y, wp.acc.z - g]).reshape([3,1]).double()
        
        # assume the yaw angle stays at 0
        b3_ref = (-acc_tensor / torch.norm(acc_tensor)).double()
        b1_yaw_tensor = torch.tensor([1., 0., 0.]).reshape([3, 1]).double()
        b2_ref = torch.cross(b3_ref, b1_yaw_tensor)
        b2_ref = b2_ref / torch.norm(b2_ref)
        b1_ref = torch.cross(b2_ref, b3_ref)
        pose = (torch.concat([b1_ref, b2_ref, b3_ref], dim=1)).double()
        R_err = torch.mm(pose, torch.t(last_ref_pose))
        q_err = rotation_matrix_2_quaternion(R_err)
        q_err = q_err / torch.norm(q_err)
        axis = torch.tensor([q_err[1], q_err[2], q_err[3]]).reshape([3, 1])
        if torch.norm(axis) != 0:
          axis = axis / torch.norm(axis)
        angle = 2 * torch.acos(q_err[0])
        angle_dot = angle / dt * axis
        angle_ddot = ((angle_dot - last_ref_angle_dot) / dt).double()

        ref_states.append((position_tensor, velocity_tensor, raw_acc_tensor, pose, angle_dot, angle_ddot))

        last_ref_pose = pose
        last_ref_angle_dot = angle_dot
        last_ref_angle_ddot = angle_ddot

    return ref_states

def compute_loss(dynamic_system, controller, controller_parameters, initial_state, ref_states, dt):
    loss = 0
    system_state = torch.clone(initial_state)
    for index, ref_state in enumerate(ref_states):
      ref_position, ref_velocity, ref_acceleration, ref_pose, ref_angular_vel, ref_angular_acc = ref_state
      controller_input = controller.get_control(controller_parameters, system_state, ref_state, None)
      system_new_state = dynamic_system.state_transition(system_state, controller_input, dt)
      
      position, pose, vel, angular_vel = system_new_state[0:3], system_new_state[3:7], system_new_state[7:10], system_new_state[10:13]

      system_state = system_new_state

      loss += torch.norm(
        ref_position - position
      )
    return loss / len(ref_states)

def func_to_get_state_error(state, ref_state):
    
    ref_position, ref_velocity, ref_acceleration, ref_pose, ref_angular_vel, ref_angular_acc = ref_state
    
    return state - torch.vstack(
       [
          ref_position,
          rotation_matrix_2_quaternion(ref_pose),
          ref_velocity,
          ref_angular_vel
       ]
    )

if __name__ == "__main__":
    g = 9.81

    # program parameters
    time_interval = 0.02
    learning_rate = 0.05
    initial_state = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([13, 1]).double()
    initial_controller_parameters = torch.tensor([[0.5003, 0.0996, 0.7173, 0.0100]]).reshape([4, 1]).double()

    quadrotor_waypoints = [
        WayPoint(0, 0, 0, 0),
        WayPoint(1, 2, -1, 2),
        WayPoint(3, 6, -5, 4),
        # WayPoint(2, 4, -3, 6),
        # WayPoint(0, 2, -1, 8),
        # WayPoint(0, 0, 0, 10),
        # WayPoint(0, 1, 3, 12)
    ]

    ref_states = get_ref_states(quadrotor_waypoints, time_interval)


    multicopter = MultiCopter(time_interval, 
                               torch.tensor(g), 
                               torch.tensor([
                                  [0.0829, 0., 0.],
                                  [0, 0.0845, 0],
                                  [0, 0, 0.1377]
                                ]))
    
    # start to tune the controller parameters
    max_tuning_iterations = 30
    tuning_times = 0
    tuner = ControllerParametersTuner(learning_rate=learning_rate)
    
    controller = SE3Controller(multicopter.m, multicopter.J)
    controller_parameters = torch.clone(initial_controller_parameters)

    # only tune positions
    states_to_tune = torch.zeros([len(initial_state), len(initial_state)])
    states_to_tune[0, 0] = 1
    states_to_tune[1, 1] = 1
    states_to_tune[2, 2] = 1
    
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
        print("Controller parameters: ", controller_parameters)
        print("Loss: ", compute_loss(multicopter, controller, controller_parameters, initial_state, ref_states, time_interval))
