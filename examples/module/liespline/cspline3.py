import torch
import os.path
import argparse
import pypose as pp
import matplotlib.pyplot as plt

def plot_result(waypoints, xrange, yrange, zrange, k = 0,
                oripoints = None, save=None, show=None):
    assert k < waypoints.shape[0]
    ax = plt.axes(projection='3d')
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_zlim(zrange)
    ax.plot3D(waypoints[k, :, 0], waypoints[k, :, 1], waypoints[k, :, 2])
    if oripoints != None:
        ax.scatter(oripoints[k, :, 0], oripoints[k, :, 1], oripoints[k, :, 2], c='r')
    if save is not None:
        file_path = os.path.join(save, 'CsplineR3.png')
        plt.savefig(file_path)
        print("Save to", file_path)
    if show:
        plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='LieSpline Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/liespline/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=True)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.save), exist_ok=True)
    print(args)
    points = torch.tensor([[[0., 0., 0.],
                            [1., 2., 2.]]])
    waypoints = pp.CSplineR3(points, 0.2)
    print(waypoints[0])
    # plot_result(waypoints, xrange, yrange, zrange,
    #             oripoints=points, show=args.show, save=args.save)
