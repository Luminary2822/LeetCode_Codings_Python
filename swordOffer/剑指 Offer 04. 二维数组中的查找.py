'''
Description: 二维数组中的查找
Author: Luminary
Date: 2021-10-06 13:17:30
LastEditTime: 2021-10-06 13:17:31
'''
class Solution:
    def findNumberIn2DArray(self, matrix, target):
        # 标志位法
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        m = len(matrix)
        n = len(matrix[0])
        # 从右上角开始遍历，往下走增大，往左走减小
        row = 0
        col = n - 1
        while row < m and col >= 0:
            if matrix[row][col] > target:
                col -= 1
            elif matrix[row][col] < target:
                row += 1
            elif matrix[row][col] == target:
                return True
        return False

        