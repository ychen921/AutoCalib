import os
import cv2
import argparse
import numpy as np
from scipy.optimize import least_squares

from Mics.LoadImages import LoadImages
from Mics.Utils import FindChessBoardCorners

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default='/home/ychen921/733/hw1/AutoCalib/Data/Calibration_Imgs', 
                        help='Default:/home/ychen921/733/hw1/AutoCalib/Data/Calibration_Imgs')
    
    Args = Parser.parse_args()
    DataPath = Args.DataPath

    color_images, gray_images, n_images = LoadImages(DataPath)
    
    # Find the chess board corners 
    H_set, imgpoints, objpoints = FindChessBoardCorners(gray_images=gray_images, color_images=color_images, n_images=n_images)
    

if __name__ == '__main__':
    main()