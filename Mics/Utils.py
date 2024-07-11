import cv2
import numpy as np

def FindChessBoardCorners(gray_images, color_images, n_images):

    #  termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((9*6, 2), np.float32)
    objp[:,:] = np.mgrid[0:9,0:6].T.reshape(-1,2)*21.5

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for i in range(n_images):
        img_gray , img_rgb = gray_images[i], color_images[i]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img_rgb, (7,6), corners2, ret)
            cv2.imwrite('Figures/Corners{}.png'.format(i), img_rgb)

    print(objp)
    
    cv2.destroyAllWindows()