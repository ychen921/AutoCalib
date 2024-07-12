import cv2
import numpy as np
import math

def FindChessBoardCorners(gray_images, color_images, n_images): 
    """
    Find corners on the chess board images and return the feature coordinates
    """
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

def Matrix_V_ij(i, j, H):
    """
    Organize matrix Vij
    """
    V_ij = np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]])

    return V_ij

def EstimateIntrinsicParameters(H_set):
    """
    Compute two fundamental constraints from homographies,
    if n images of model plane are observed, by stacking n 
    such homogeneour equations in b to sovle Vb = 0.
    """
    MatrixV = []
    for i in range(len(H_set)):
        H = H_set[i]
        V_12 = Matrix_V_ij(i=0, j=1, H=H)
        SubV_11_22 = Matrix_V_ij(i=0, j=0, H=H) - Matrix_V_ij(i=1, j=1, H=H)

        MatrixV.append(V_12)
        MatrixV.append(SubV_11_22)

    MatrixV = np.array(MatrixV)
    _, _, Vt = np.linalg.svd(MatrixV)
    b = Vt[-1]
    print('\nB matrix: ')
    print(b.T)

    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]

    # Extraction of the Intrinsic Parameters from Matrix B
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lamda = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    alpha = math.sqrt(lamda / B11)
    beta = math.sqrt(lamda*B11 / (B11*B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lamda
    u0 = gamma * v0 / beta - B13 * alpha**2 / lamda

    # Intrisic matrix K = [alpha, gamma, u0]
    #                     [  0  ,  beta, v0]
    #                     [  0  ,   0  ,  1]

    K = np.array([[alpha, gamma, u0],
                  [0, beta, v0],
                  [0, 0, 1]
                  ])
    
    return K