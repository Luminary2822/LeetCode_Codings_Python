# 旋转图像
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        # 本题不用return，直接修改matrix
        n = len(matrix)
        matrix_new = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix_new[j][n-i-1] = matrix[i][j]
        matrix[:] = matrix_new

        # 第二种方法：充分利用库函数的便捷方法：直接修改matrix，利用numpy顺时针-1旋转90度函数
        """
        import numpy as np
        matrix[:]=np.rot90(matrix,-1).tolist()
        """

        # 第三种方法：先矩阵转置再水平镜像翻转即可
        """
        # 矩阵转置
        matrix_trans = np.transpose(matrix)
        # 矩阵镜像fliplr水平方向，flipud垂直方向
        res = np.fliplr(matrix_trans)
        """