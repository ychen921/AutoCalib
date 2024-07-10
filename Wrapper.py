import os
import cv2
import argparse
import numpy as np
from scipy.optimize import least_squares
from Mics.LoadImages import LoadImages

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath',  
                        default='/home/ychen921/733/hw1/AutoCalib/Data/Calibration_Imgs', help='Default:/home/ychen921/733/hw1/AutoCalib/Data/Calibration_Imgs')
    
    Args = Parser.parse_args()
    DataPath = Args.DataPath

    gray_scale, original = LoadImages(DataPath)
    


if __name__ == '__main__':
    main()