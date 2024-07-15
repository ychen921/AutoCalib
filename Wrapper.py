import os
import cv2
import argparse
import numpy as np
from scipy.optimize import least_squares

from Mics.LoadImages import LoadImages
from Mics.Utils import FindChessBoardCorners, EstimateIntrinsicParameters
from Mics.Utils import EstimateExtrinsicMatrix, ReprojectionError

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default='/home/ychen921/733/hw1/AutoCalib/Data/Calibration_Imgs', 
                        help='Default:/home/ychen921/733/hw1/AutoCalib/Data/Calibration_Imgs')
    
    Args = Parser.parse_args()
    DataPath = Args.DataPath

    color_images, gray_images, n_images = LoadImages(DataPath)
    
    # Find the chess board corners 
    H_set, imgpoints, objpoints = FindChessBoardCorners(gray_images=gray_images, color_images=color_images, n_images=n_images)
    
    # Solving for approximate camera intrinsic matrix K
    Init_K = EstimateIntrinsicParameters(H_set)
    print('\nApproximate camera intrinsic matrix K:')
    print(Init_K)

    # Estimate approximate camera extrinsics (R and t)
    # and compute reprojection error
    Extrinsics = []
    reproj_errors = []
    for i in range(n_images):
        H = H_set[i]
        img_pts = imgpoints[i]
        R_t = EstimateExtrinsicMatrix(K_init=Init_K, H=H)
        Extrinsics.append(R_t)

        Error = ReprojectionError(img_pts=img_pts, obj_pts=objpoints, R_t=R_t, K_init=Init_K)
        reproj_errors.append(Error)

if __name__ == '__main__':
    main()