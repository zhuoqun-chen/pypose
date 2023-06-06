import torch

# accept column vector
def hat(vector):
    device = vector.device
    zero_t = torch.tensor([0.], device=device)
    return torch.squeeze(torch.stack([
        torch.stack([zero_t, -vector[2], vector[1]], dim=-1),
        torch.stack([vector[2], zero_t, -vector[0]], dim=-1),
        torch.stack([-vector[1], vector[0], zero_t], dim=-1)
    ]))

def vee(skew_symmetric_matrix):
    return torch.stack(
        [-torch.unsqueeze(skew_symmetric_matrix[1, 2], dim=0),
        torch.unsqueeze(skew_symmetric_matrix[0, 2], dim=0),
        -torch.unsqueeze(skew_symmetric_matrix[0, 1], dim=0)])

# accept column vector
def quaternion_2_rotation_matrix(q):
    device = q.device
    q = q / torch.norm(q)
    qahat = hat(q[1:4])
    return (torch.eye(3, device=device)
        + 2 * torch.mm(qahat, qahat) + 2 * q[0] * qahat).double()

def rotation_matrix_2_quaternion(R):
    device = R.device
    tr = R[0,0] + R[1,1] + R[2,2];

    if tr > 0:
        S = torch.sqrt(tr+1.0) * 2
        qw = 0.25 * S;
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S

    q = torch.stack([
        torch.tensor([qw], device=device),
        torch.tensor([qx], device=device),
        torch.tensor([qy], device=device),
        torch.tensor([qz], device=device)])
    q = q * ((qw+0.00000001) / torch.abs(qw + 0.00000001))
    q = q / torch.norm(q)

    return q

def get_shortest_path_between_angles(original_ori, des_ori):
    e_ori = des_ori - original_ori
    if abs(e_ori) > torch.pi:
        if des_ori > original_ori:
            e_ori = - (original_ori + 2 * torch.pi - des_ori)
        else:
            e_ori = des_ori + 2 * torch.pi - original_ori
    return e_ori

def get_desired_angular_speed(original_ori, des_ori, dt):
    return get_shortest_path_between_angles(original_ori, des_ori) / dt

def angular_vel_2_quaternion_dot(quaternion, w):
    device = quaternion.device
    p, q, r = w
    zero_t = torch.tensor([0.], device=device)
    return -0.5 * torch.mm(torch.squeeze(torch.stack(
        [
            torch.stack([zero_t, p, q, r], dim=-1),
            torch.stack([-p, zero_t, -r, q], dim=-1),
            torch.stack([-q, r, zero_t, -p], dim=-1),
            torch.stack([-r, -q, p, zero_t], dim=-1)
        ])), quaternion)
