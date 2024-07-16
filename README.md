# AutoCalib
In this project, we implement an automatic method to perform efficient camera calibration which is proposed by Zhengyou Zhang of Microsoft in [this paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) [1].

Recall that the camera calibration matrix **K** is given as follows:

$$
K = \begin{bmatrix} 
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1\\
\end{bmatrix}
$$

and radial distortion parameters are denoted by $$k_1$$ and $$k_2$$ respectively. Our task is to estimate $$f_x, f_y, c_x, c_y, k_1, k_2$$.

## Data
The paper [1] relies on a calibration target (checkerboard in our case) to estimate camera intrinsic parameters. The checkboard target was printed on an A4 paper and the the size of each square was 21.5mm. Note that the Y axis has an odd number of squares and the X axis has an even number of squares. Thirteen images of the checkerboard were taken from a Google Pixel XL phone with focus locked.

## Calibration Procedure
First, we compute the homography between the model plane and its image. Then, refer to Section 3.1 and Appendix B in the paper [1]. We can solve the approximate camera intrinsic matrix by computing the matrix **B** and extracting the intrinsic parameters.

From here, since we have the initial calibration matrix **K**, the camera extrinsic matrix (rotation and translation) can be calculated by its homography and calibration matrix. Because we assumed that the camera has minimal distortion we can assume that $$k_c=[0,0]^T$$ for a good initial estimate.

Up to now

## References
1. Zhang, Z. (2000). A flexible new technique for camera calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11), 1330-1334. 
