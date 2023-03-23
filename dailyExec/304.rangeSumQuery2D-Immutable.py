#  前缀二维矩阵：preSum[i][j]preSum[i][j] 表示 从 [0,0][0,0] 位置到 [i,j][i,j] 位置的子矩形所有元素之和。
class NumMatrix(object):
    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        if not matrix or not matrix[0]:
            M,N = 0,0
        else:
            M = len(matrix)
            N = len(matrix[0])
        # 初始化前缀矩阵，比原矩阵多一行一列：M+1行N+1列
        self.preSum = [[0]*(N+1) for _ in range(M+1)]
        # 前缀矩阵求解公式：preSum[i][j]=preSum[i−1][j]+preSum[i][j−1]−preSum[i−1][j−1]+matrix[i][j]
        for i in range(M):
            for j in range(N):
                self.preSum[i+1][j+1] = self.preSum[i][j+1] + self.preSum[i+1][j] - self.preSum[i][j] + matrix[i][j]

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        # 利用前缀矩阵求子矩形面积
        return self.preSum[row2+1][col2+1] - self.preSum[row2+1][col1] - self.preSum[row1][col2+1] + self.preSum[row1][col1]
        