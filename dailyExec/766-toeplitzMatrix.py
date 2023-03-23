class Solution(object):
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        # 思路：每一个位置都要跟其右下角的元素相等，相邻行列表错位相等
        for i in range(len(matrix)-1):
            # 切片操作：[:-1]去除最后一位；[1:]从第二位开始，i行和i+1行错位列表相等
                if matrix[i][:-1] != matrix[i+1][1:]:
                    return False
        return True
                