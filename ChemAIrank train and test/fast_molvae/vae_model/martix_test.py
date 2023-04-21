# 导入numpy库
import numpy as np

# 定义一个矩阵
matrix = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]])

# 尝试对矩阵进行Cholesky分解
try:
    result = np.linalg.cholesky(matrix)
    print("The matrix is positive definite.")
except np.linalg.LinAlgError:
    print("The matrix is not positive definite.")