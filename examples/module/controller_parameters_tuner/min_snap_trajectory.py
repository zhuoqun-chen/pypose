import numpy as np
import sys
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

class Vector3d:
  def __init__(self, x=0, y=0, z=0):
    self.x = x
    self.y = y
    self.z = z

class WayPoint:
  def __init__(self, px, py, pz=0, timestamp=0):
    self.position = Vector3d()
    self.vel = Vector3d()
    self.acc = Vector3d()
    self.ts = 0.0;
    self.position.x = px
    self.position.y = py
    self.position.z = pz
    self.ts = timestamp

  def setTime(self, timestamp):
    self.ts = timestamp;

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
    A = np.zeros((A_dimemsion, A_dimemsion), dtype=np.double)
    b_pos_x = np.zeros((A_dimemsion, 1), dtype=np.double)
    b_pos_y = np.zeros((A_dimemsion, 1), dtype=np.double)
    b_pos_z = np.zeros((A_dimemsion, 1), dtype=np.double)

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
      self.set_derivative_constraint(A, wps[0].ts, 1, wps[0], poly_order, deri_order, A_dimemsion - 1 - 2 * max_deri + deri_order, True)
      self.set_derivative_constraint(A, wps[N-1].ts, N-1, wps[N-1], poly_order, deri_order, A_dimemsion - 1 - max_deri + deri_order, False)

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

    A_inv = np.linalg.pinv(A)
    co_x = A_inv @ b_pos_x
    co_y = A_inv @ b_pos_y
    co_z = A_inv @ b_pos_z

    traj_wps = []
    accumulated_ts = 0
    curr_traj_index = 1

    while accumulated_ts + dt <= wps[N-1].ts + dt:
      if accumulated_ts > wps[curr_traj_index].ts:
        curr_traj_index += 1
        accumulated_ts = wps[curr_traj_index - 1].ts
      last_wp = wps[curr_traj_index - 1]
      curr_wp = wps[curr_traj_index]

      wp = WayPoint(0, 0)
      wp.setTime(accumulated_ts)

      poly_pos_t = []
      poly_vel_t = []
      poly_acc_t = []
      for index in range(0, poly_order + 1):
        poly_pos_t.append(pow(accumulated_ts, index))
        poly_vel_t.append(0 if (index - 1) < 0 else (np.math.factorial(index) / np.math.factorial(index - 1) * pow(accumulated_ts, index - 1)))
        poly_acc_t.append(0 if (index - 2) < 0 else (np.math.factorial(index) / np.math.factorial(index - 2) * pow(accumulated_ts, index - 2)))

      wp.position.x = (poly_pos_t @ co_x[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index])
      wp.position.y = poly_pos_t @ co_y[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index]
      wp.position.z = poly_pos_t @ co_z[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index]
      wp.vel.x = poly_vel_t @ co_x[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index]
      wp.vel.y = poly_vel_t @ co_y[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index]
      wp.vel.z = poly_vel_t @ co_z[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index]
      wp.acc.x = poly_acc_t @ co_x[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index]
      wp.acc.y = poly_acc_t @ co_y[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index]
      wp.acc.z = poly_acc_t @ co_z[(poly_order + 1) * (curr_traj_index - 1) : (poly_order + 1) * curr_traj_index]
      traj_wps.append(wp)

      accumulated_ts += dt

    return traj_wps

  # traj_index start from 1
  def set_position_constraint(self, A, traj_index, last_wp, curr_wp, poly_order, row_index):
    poly_order += 1
    traj_index -= 1
    for i in range(0, poly_order):
      A[row_index, poly_order * traj_index + i] = 1 * pow(last_wp.ts, i)
      A[row_index + 1, poly_order * traj_index + i] = 1 * pow(curr_wp.ts, i)

  def set_continus_derivative_constraint(self, A, traj_index, last_wp, curr_wp, poly_order, derivative_order, row_index):
    poly_order += 1
    traj_index -= 1
    for i in range(0, poly_order):
      A[row_index, poly_order * traj_index + i] = 0 if (i - derivative_order) < 0 else (np.math.factorial(i) / np.math.factorial(i - derivative_order)) * pow(curr_wp.ts, i - derivative_order)
      A[row_index, poly_order * traj_index + poly_order + i] = -(np.math.factorial(i) / np.math.factorial(i - derivative_order)) * pow(curr_wp.ts, i - derivative_order) if (i - derivative_order) >= 0 else 0

  def set_derivative_constraint(self, A, ts, traj_index, wp, poly_order, derivative_order, row_index, is_sub_traj_start):
    poly_order += 1
    traj_index -= 1
    for i in range(0, poly_order):
      if is_sub_traj_start:
        A[row_index, poly_order * traj_index + i] = (np.math.factorial(i) / np.math.factorial(i - derivative_order)) * pow(ts, i - derivative_order) if (i - derivative_order) >= 0 else 0
      else:
        A[row_index, poly_order * traj_index + i] = 0 if (i - derivative_order) < 0 else (np.math.factorial(i) / np.math.factorial(i - derivative_order)) * pow(ts, i - derivative_order)

  def get_desired_states_in_2d(self, traj, dt):
    pass

  def get_desired_states_in_3d(self):
    pass

def plot_3d_trajectory(wps):
    xs = []
    ys = []
    zs = []
    vel_x = []
    vel_y = []
    vel_z = []
    acc_x = []
    acc_y = []
    acc_z = []
    dts = []
    f, ax = plt.subplots(nrows=3, ncols=1)
    for wp in wps:
      xs.append(wp.position.x)
      ys.append(wp.position.y)
      zs.append(wp.position.z)
      vel_x.append(wp.vel.x)
      vel_y.append(wp.vel.y)
      vel_z.append(wp.vel.z)
      acc_x.append(wp.acc.x)
      acc_y.append(wp.acc.y)
      acc_z.append(wp.acc.z)
      dts.append(wp.ts)

    ax[0].plot(dts, xs, label='x')
    ax[0].plot(dts, ys, label='y')
    ax[0].plot(dts, zs, label='z')
    ax[1].plot(dts, vel_x, label='x')
    ax[1].plot(dts, vel_y, label='y')
    ax[1].plot(dts, vel_z, label='z')
    ax[2].plot(dts, acc_x, label='x')
    ax[2].plot(dts, acc_y, label='y')
    ax[2].plot(dts, acc_z, label='z')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
waypoints = [
  WayPoint(0, 0, 1, 0),
  WayPoint(1, 2, 0, 1),
  WayPoint(9, 3, 0, 20),
]
traj_gen = PolynomialTrajectoryGenerator()
traj_gen.assign_timestamps_in_waypoints(waypoints)
desired_states = traj_gen.generate_trajectory(waypoints, 0.1, 7)

plot_3d_trajectory(desired_states)
