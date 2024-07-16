import argparse
import numpy as np

from Mics.LoadImages import LoadImages
from Mics.Utils import FindChessBoardCorners, EstimateIntrinsicParameters
from Mics.Utils import EstimateExtrinsicMatrix, Optimization
from Mics.Utils import ReprojectionError, ReprojectionErrorDistort
from Mics.Utils import Visualization


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

        # Compute camera intrinsic
        R_t = EstimateExtrinsicMatrix(K_init=Init_K, H=H)
        Extrinsics.append(R_t)

        # Compute reprojection from camera intrinsic
        Error = ReprojectionError(img_pts=img_pts, obj_pts=objpoints, R_t=R_t, K_init=Init_K)
        reproj_errors.append(Error)

    print('\nMean Reprojection Error before Optimization:')
    print(np.mean(reproj_errors), np.std(reproj_errors))
    
    print('\n########################################################################')
    print('Optimizing calibration matrix K by using Levenberg-Marquardt Algorithm...')
    print('########################################################################')
    K_optim, k1, k2 = Optimization(K=Init_K, H_set=H_set, img_pts=imgpoints, obj_pts=objpoints)

    reproj_errors = []
    reproj_points = []
    for i in range(n_images):
        H = H_set[i]
        img_pts = imgpoints[i]

        # Compute camera intrinsic
        R_t = EstimateExtrinsicMatrix(K_init=K_optim, H=H)

        # Compute reprojection from optimized camera intrinsic
        Error, pts = ReprojectionErrorDistort(img_pts=img_pts, obj_pts=objpoints, R_t=R_t, K=K_optim, k1=k1, k2=k2)
        reproj_errors.append(Error)
        reproj_points.append(pts)
    
    print('\nMean Reprojection Error after Optimization:')
    print(np.mean(reproj_errors), np.std(reproj_errors))

    print('\nCalibration matrix K after Optimization:')
    print(K_optim)

    print("\nDistortion coefficients after optimization are: ")
    print(k1, k2)

    Visualization(imgpoints=imgpoints, reproj_points=reproj_points, images=color_images)

if __name__ == '__main__':
    main()