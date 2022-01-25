import cv2
import numpy as np
from pathlib import Path


def alg():
    H = np.ones((row, col, D))
    idxs = np.tri(col, D)
    idx = np.tile(idxs, (row, 1, 1))
    H = H * idx
    #X, Y, Z = np.ix_(np.arange(row), np.arange(col), np.arange(D))
    #eiges = np.fromfunction(lambda i, j: np.abs(i - j), (D, D))
    weight = np.fromfunction(lambda i, j, d, d_: np.sqrt((left[i, j] - right[i, j - d])**2) + alpha * np.sqrt((d - d_)**2), (row, col, D, D), dtype=int)
    H = np.where(H == 0, np.inf, 0)
    #H = np.where(H == 0, np.inf, np.min(np.abs(left[X, Y] - right[X, Y - Z])))
    #place = np.ones((row, col))
    #place[:, 0] = np.min(H[:, 0, :], axis=1)
    #print(place)
    #d = np.zeros(col, dtype=int)
    H[:, 0, 0] = weight[:, 0, 0, 0]
    d = np.zeros((row, col), dtype=int)
    for i in range(1, col-1):
        #k = weight[:, i, :, d[:, i]]
        k = np.fromfunction(lambda r, y: weight[r, i, y, d[r, i]], (row, D), dtype=int)
        H[:, i, :] = np.where(H[:, i, :] != np.inf, k, np.inf)
        d[:, i+1] = np.argmin(H[:, i, :], axis=1)

       # place[:, i] = np.min(H[:, i, :] + np.tile(alpha * abs(eiges[d[i], :]), (row, 1)), axis=1)
        #d[i+1] = np.argmin(H[:, i, :] + np.tile(alpha * abs(eiges[d[i], :]), (row, 1)))
    H[:, col-1, :] = np.where(H[:, col-1, :] != np.inf, weight[:, col-1, :, d[:, i]][3], np.inf)
    #place[:, col-1] = np.min(H[:, col-1, :] + np.tile(alpha * abs(eiges[d[col-1], :]), (row, 1)), axis=1)
    final = (d - np.min(d))/(np.max(d) - np.min(d))
    #print(final)
    cv2.namedWindow("final")
    cv2.imshow("final", final)
    cv2.waitKey(0)

D = 10
alpha = 3
file_path_r = Path("im0.ppm").resolve()
left = cv2.imread(str(file_path_r), cv2.IMREAD_GRAYSCALE)
row, col = left.shape[:2]

file_path_l = Path("im1.ppm")
right = cv2.imread(str(file_path_l), cv2.IMREAD_GRAYSCALE)

alg()

cv2.destroyAllWindows()




