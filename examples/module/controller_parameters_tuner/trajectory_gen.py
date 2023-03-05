import numpy as np
import torch

from waypoint import WayPoint
from commons import get_tensor_item, get_shortest_path_between_angles

class PolynomialTrajectoryGenerator:
  def __init__(self):
    return
  
  def assign_timestamps_in_waypoints(self, wps, distance_scala_factor = 1):
      accmu_t = 0;
      for index in range(1, len(wps)):
        last_wp = wps[index - 1]
        curr_wp = wps[index]
        dis = np.sqrt(
            pow(curr_wp.position.x - last_wp.position.x, 2)
            + pow(curr_wp.position.y - last_wp.position.y, 2)
            + pow(curr_wp.position.z - last_wp.position.z, 2)
          )
        accmu_t += dis * distance_scala_factor
        wps[index].setTime(accmu_t)

  # The initial and final velocity are both 0
  # return traj_points
  def generate_trajectory(self, wps, dt, poly_order=3):
      if poly_order < 3 or poly_order %2 == 0:
          return -1

      N = len(wps)
      A_dimemsion = (poly_order + 1) * (N-1) 
      A = np.zeros((A_dimemsion, A_dimemsion))
      b_pos_x = np.zeros((A_dimemsion, 1))
      b_pos_y = np.zeros((A_dimemsion, 1))
      b_pos_z = np.zeros((A_dimemsion, 1))

      # set constraints of each point's position
      for index in range(1, len(wps)):
          last_wp = wps[index - 1]
          curr_wp = wps[index]

          start_row_index = (poly_order + 1) * (index - 1)
          # for each sub trajectory, set position constraint
          self.set_position_constraint(A, index, last_wp, curr_wp, poly_order, start_row_index)

          if index < len(wps) - 1:
            # for each sub trajectory, set continuous velocity and acceleration constraint
              for deri_order in range(1, poly_order):
                  self.set_continus_derivative_constraint(A, index, last_wp, curr_wp, poly_order, deri_order, start_row_index + 1 + deri_order)

      # set initial and final velocity constraints
      max_deri = int(poly_order / 2)
      for deri_order in range(1, max_deri + 1):
          self.set_derivative_constraint(A, 1, wps[0], poly_order, deri_order, A_dimemsion - 1 - 2 * max_deri + deri_order, True)
          self.set_derivative_constraint(A, N-1, wps[N-1], poly_order, deri_order, A_dimemsion - 1 - max_deri + deri_order, False)

      # construst b
      for index in range(1, len(wps)):
          last_wp = wps[index - 1]
          curr_wp = wps[index]
          start_row_index = (2 + max_deri * 2) * (index - 1)
          b_pos_x[start_row_index][0] = last_wp.position.x
          b_pos_x[start_row_index + 1][0] = curr_wp.position.x
          b_pos_y[start_row_index][0] = last_wp.position.y
          b_pos_y[start_row_index + 1][0] = curr_wp.position.y
          b_pos_z[start_row_index][0] = last_wp.position.z
          b_pos_z[start_row_index + 1][0] = curr_wp.position.z
      
      A_inv = np.linalg.inv(A)
      co_x = A_inv @ b_pos_x
      co_y = A_inv @ b_pos_y
      co_z = A_inv @ b_pos_z

      traj_wps = []
      accumulated_ts = 0
      curr_traj_index = 1
      while accumulated_ts <= wps[N-1].ts:
          last_wp = wps[curr_traj_index - 1]
          curr_wp = wps[curr_traj_index]

          wp = WayPoint(0, 0)
          wp.setTime(accumulated_ts)
          ot = curr_wp.ts - last_wp.ts
          t = (accumulated_ts - last_wp.ts) / ot

          poly_pos_t = []
          poly_vel_t = []
          poly_acc_t = []
          for index in range(0, poly_order + 1):
              poly_pos_t.append(pow(t, index))
              poly_vel_t.append(0 if (index - 1) < 0 else (np.math.factorial(index) / np.math.factorial(index - 1) * pow(t, index - 1)))
              poly_acc_t.append(0 if (index - 2) < 0 else (np.math.factorial(index) / np.math.factorial(index - 2) * pow(t, index - 2)))
          wp.position.x = poly_pos_t @ co_x[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index, 0]
          wp.position.y = poly_pos_t @ co_y[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index, 0]
          wp.position.z = poly_pos_t @ co_z[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index, 0]
          wp.vel.x = poly_vel_t @ co_x[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index, 0] / ot
          wp.vel.y = poly_vel_t @ co_y[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index, 0] / ot
          wp.vel.z = poly_vel_t @ co_z[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index, 0] / ot
          wp.acc.x = poly_acc_t @ co_x[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index, 0] / pow(ot, 2)
          wp.acc.y = poly_acc_t @ co_y[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index, 0] / pow(ot, 2)
          wp.acc.z = poly_acc_t @ co_z[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index, 0] / pow(ot, 2)

          if accumulated_ts + dt > wps[curr_traj_index].ts:
              curr_traj_index += 1
          accumulated_ts += dt
          traj_wps.append(wp)
      return traj_wps

  # traj_index start from 1
  def set_position_constraint(self, A, traj_index, last_wp, curr_wp, poly_order, row_index):
      poly_order += 1
      traj_index -= 1
      A[row_index, poly_order * traj_index : poly_order * traj_index + poly_order] = np.zeros((1, poly_order))
      A[row_index, poly_order * traj_index] = 1
      A[row_index + 1, poly_order * traj_index : poly_order * traj_index + poly_order] = np.ones((1, poly_order))

  def set_continus_derivative_constraint(self, A, traj_index, last_wp, curr_wp, poly_order, derivative_order, row_index):
    poly_order += 1
    traj_index -= 1
    for i in range(0, poly_order):
        A[row_index, poly_order * traj_index + i] = 0 if (i - derivative_order) < 0 else (np.math.factorial(i) / np.math.factorial(i - derivative_order))
        A[row_index, poly_order * traj_index + poly_order + i] = -(np.math.factorial(i) / np.math.factorial(i - derivative_order)) if (i - derivative_order) == 0 else 0

  def set_derivative_constraint(self, A, traj_index, wp, poly_order, derivative_order, row_index, is_sub_traj_start):
    poly_order += 1
    traj_index -= 1
    for i in range(0, poly_order):
        if is_sub_traj_start:
            A[row_index, poly_order * traj_index + i] = (np.math.factorial(i) / np.math.factorial(i - derivative_order)) if (i - derivative_order) == 0 else 0
        else:
            A[row_index, poly_order * traj_index + i] = 0 if (i - derivative_order) < 0 else (np.math.factorial(i) / np.math.factorial(i - derivative_order))

  def get_desired_states_in_2d(self, traj, dt):
    car_desired_states = []
    last_desired_state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    for wp in traj:
        last_desired_orientation, \
        last_desired_orientation_dot, \
        last_desired_orientation_ddot = last_desired_state[-3:]
        current_orientation = torch.atan2(torch.tensor(wp.vel.y), torch.tensor(wp.vel.x)).item()
        last_desired_orientation_remaindar = torch.atan2(torch.sin(torch.tensor(last_desired_orientation)), torch.cos(torch.tensor(last_desired_orientation))).item()
        delta_angle = get_shortest_path_between_angles(last_desired_orientation_remaindar, current_orientation)
        current_orientation = last_desired_orientation + delta_angle
        current_ori_dot = (current_orientation - last_desired_orientation) / dt
        
        car_desired_states.append(
          (wp.position.x, \
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

  def get_desired_states_in_3d(self):
    pass

  r'''
      generate circular waypoints with constant velocity and angular speed
  '''
  def generate_circle_waypoints(self, radius, constant_velocity, dt):
      perimeter = torch.pi * 2 * radius
      total_time = perimeter / constant_velocity
      desired_waypoints = []

      accu_ts = 0
      while accu_ts < total_time:
          theta = accu_ts / total_time * 2 * torch.pi
          theta_tensor = torch.tensor(theta)
          half_pi_tensor = torch.tensor(0.5 * torch.pi)
          wp = WayPoint(
            radius * torch.cos(theta_tensor - half_pi_tensor),
            radius * torch.sin(theta_tensor - half_pi_tensor) + radius,
            0,
            accu_ts
          )
          wp.vel.x = torch.cos(torch.tensor(theta)) * constant_velocity
          wp.vel.y = torch.sin(torch.tensor(theta)) * constant_velocity
          wp.vel.z = 0
          wp.acc.x = 0
          wp.acc.y = 0
          wp.acc.z = 0

          desired_waypoints.append((
            wp.position.x,
            wp.position.y,
            wp.vel.x,
            wp.vel.y,
            wp.acc.x,
            wp.acc.y,
            theta,
            2 * torch.pi / total_time,
            0
          ))

          accu_ts += dt
    
      return desired_waypoints