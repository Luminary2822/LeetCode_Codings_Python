class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        # 思路：循环【取矩阵第一行，逆时针旋转九十度】
        import numpy as np
        res = []
        while matrix:
            res += matrix.pop(0)  #每次提取第一排元素
            # zip函数+列表逆序：将剩余的元素进行逆时针旋转九十度
            # *matrix先解压，然后再用zip重新组合，逆序
            matrix = list(zip(*matrix))[::-1]   
            # matrix = [(6,9),(5,8),(4,7)]
            # numpy逆时针旋转函数
            # matrix[:]=np.rot90(matrix,1).tolist()
        return res

        # 第二种方法：按照题意理解法
        # 规则：如果当前行（列）遍历结束之后，就需要把这一行（列）的边界向内移动一格
        """
        if not matrix or not matrix[0]: return []
        M, N = len(matrix), len(matrix[0])
        # 边界
        left, right, up, down = 0, N - 1, 0, M - 1
        res = []
        x, y = 0, 0
        # 移动方向
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        cur_d = 0
        while len(res) != M * N:
            res.append(matrix[x][y])
            if cur_d == 0 and y == right:
                cur_d += 1
                up += 1
            elif cur_d == 1 and x == down:
                cur_d += 1
                right -= 1
            elif cur_d == 2 and y == left:
                cur_d += 1
                down -= 1
            elif cur_d == 3 and x == up:
                cur_d += 1
                left += 1
            cur_d %= 4
            x += dirs[cur_d][0]
            y += dirs[cur_d][1]
        return res
        """