class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        # 空间复杂度O(m+n)的解法：记录每行每列是否出现0
        m = len(matrix)
        n = len(matrix[0])
        # 标记数组记录0位置出现的行列
        row,col = [False] * m,[False] * n

        for i in range(m):
            for j in range(n):
                # 出现0的位置将其行列都标注为True
                if matrix[i][j] == 0:
                    row[i] = col[j] = True
        for i in range(m):
            for j in range(n):
                # 二次遍历将之前标注为True的行列置为0
                if row[i] or col[j]:
                    matrix[i][j] = 0

        # 空间复杂度O(1)的解法：使用第 0 行和第 0 列来保存 matrix[1:M][1:N] 中是否出现了 0
        """
        if not matrix or not matrix[0]:
            return
        row0,col0 = False, False
        M, N = len(matrix),len(matrix[0])
        # 1. 统计第一行和第一列是否有0
        for i in range(M):
            if matrix[i][0] == 0:
                col0 = True
        for j in range(N):
            if matrix[0][j] == 0:
                row0 = True
        # 2.开始遍历矩阵，遇到0标记到对应的第一行和第一列
        for i in range(1, M):
            for j in range(1, N):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        
        # 3.看第一行和第一列哪位元素为0.将对应的行和列元素全部标记位0
        for i in range(1, M):
            for j in range(1, N):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        # 4.最后根据第一行和第一列的标记位判断是否存在0元素，有的话将其行列置为0
        if row0:
            for j in range(N):
                matrix[0][j] = 0
        if col0:
            for i in range(M):
                matrix[i][0] = 0
        """
        
a = Solution()
print(a.setZeroes([[1,1,1],[1,0,1],[1,1,1]]))