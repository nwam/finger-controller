"""
This is a bandage module for fixing errors in recording data.
All it does is set the camera side of a pickle from a recording session.
"""
import argparse
import pickle
from recording import CamSide, CamProps

def set_cam_side(path, cam_side):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        data[0].side = cam_side

    pickle.dump(data, open(path, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('camera_side', type=str,
            help='The side of the user where the camera is placed')
    args = parser.parse_args()
    cam_side = args.camera_side.lower()
    if cam_side == 'left' or cam_side == 'l':
        cam_side = CamSide.LEFT
    elif cam_side == 'right' or cam_side == 'r':
        cam_side = CamSide.RIGHT
    else:
        print('Invalid camera side. Please use l or r.')
        exit()

    set_cam_side(args.path, cam_side)
