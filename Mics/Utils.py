import cv2
import numpy as np

def FindChessBoardCorners(gray_images, color_images, n_images):

    #  termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = np.zeros((9*6, 2), np.float32)
    objpoints[:,:] = np.mgrid[0:9,0:6].T.reshape(-1,2)*21.5 # 3d point in real world space

    # Arrays to store image points from all the images.
    imgpoints = [] # 2d points in image plane.
    Hs = []
    for i in range(n_images):
        img_gray , img_rgb = gray_images[i], color_images[i]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img_rgb, (9,6), corners2, ret)
            # cv2.imwrite('Figures/Corners{}.png'.format(i), img_rgb)

            H, _ = cv2.findHomography(objpoints[:20], corners2[:20])
            Hs.append(H)
    
    cv2.destroyAllWindows()

    return Hs, imgpoints, objpoints

def Solve_V_ij(i, j, H):
    V_ij = np.array([
        [H[0, i] * H[0, j]],
        [H[0, i] * H[1, j] + H[1, i] * H[0, j]],
        [H[1, i] * H[1, j]],
        [H[2, i] * H[0, j] + H[0, i] * H[2, j]],
        [H[2, i] * H[1, j] + H[1, i] * H[2, j]],
        [H[2, i] * H[2, j]]
    ])

def EstimateIntrinsicParameters(H_set):
    pass