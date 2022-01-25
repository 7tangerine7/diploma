import cv2
import numpy as np
from pathlib import Path


def alg():
    H = np.ones((row, col, D))
    idxs = np.tri(col, D)
    idx = np.tile(idxs, (row, 1, 1))
    H = H * idx
    weight = np.fromfunction(lambda i, j, d, d_: np.sqrt((left[i, j] - right[i, j - d])**2) + alpha * np.sqrt((d - d_)**2), (row, col, D, D), dtype=int)
    H = np.where(H == 0, np.inf, 0)
    H[:, 0, 0] = weight[:, 0, 0, 0]
    d = np.zeros((row, col), dtype=int)
    for i in range(1, col-1):
        k = np.fromfunction(lambda r, y: weight[r, i, y, d[r, i]], (row, D), dtype=int)
        H[:, i, :] = np.where(H[:, i, :] != np.inf, k, np.inf)
        d[:, i+1] = np.argmin(H[:, i, :], axis=1)
    H[:, col-1, :] = np.where(H[:, col-1, :] != np.inf, weight[:, col-1, :, d[:, i]][3], np.inf)
    final = (d - np.min(d))/(np.max(d) - np.min(d))
    cv2.namedWindow("final")
    cv2.imshow("final", final)
    cv2.waitKey(0)

D = 12
alpha = 3
file_path_r = Path("im0.ppm").resolve()
left = cv2.imread(str(file_path_r), cv2.IMREAD_GRAYSCALE)
row, col = left.shape[:2]

file_path_l = Path("im2.ppm")
right = cv2.imread(str(file_path_l), cv2.IMREAD_GRAYSCALE)

alg()

cv2.destroyAllWindows()




