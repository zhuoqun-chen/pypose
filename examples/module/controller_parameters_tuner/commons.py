import torch
from pypose.lietensor.basics import vec2skew

# accept column vector
def quaternion_2_rotation_matrix(q):
    device = q.device
    q = q / torch.norm(q)
    qahat = torch.squeeze(vec2skew(torch.t(q[1:4])))
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
