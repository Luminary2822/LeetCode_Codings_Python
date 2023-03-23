class Solution(object):
    def transpose(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
    # 本题的矩阵的行列数可能不等，因此不能做原地操作，需要新建数组
    # 获取矩阵的行和列
    M,N = len(matrix),len(matrix[0])
    # 新建一个N行M列的矩阵
    res = [[0]*M for i in range(N)]
    for i in range(M):
        for j in range(N):
            res[j][i] = matrix[i][j]
    return res