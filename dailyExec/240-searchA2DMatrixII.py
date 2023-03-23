'''
Description: 搜索二维矩阵II
Author: Luminary
Date: 2021-10-25 20:13:32
LastEditTime: 2021-10-25 20:50:32
'''

class Solution:
    def searchMatrix(self, matrix, target) :
        # 标志位：可以选择从左下角或者右上角开始遍历
        # 从左下角开始，左下角元素是这一行中最小，这一列中最大
        m = len(matrix)
        n = len(matrix[0])
        if m == 0 or n == 0:
            return False
        
        i = m - 1
        j = 0
        while i >= 0 and j < n:
            # 左下角等于目标则找到
            if matrix[i][j] == target:
                return True
            # 左下角元素小于目标，目标元素不可能在当前列，向右走规模可在去掉第一列的子矩阵中寻找
            elif matrix[i][j] < target:
                j = j + 1
            # 左下角元素大于目标，目标元素不可能在当前行，向上走规模可在去掉最后一行的子矩阵中寻找
            else:
                i = i - 1
        return False

        # 右上角法
        m = len(matrix)
        n = len(matrix[0])
        if m == 0 or n == 0:
            return False
        i = 0
        j = n - 1
        while i < m and j >= 0:
            if matrix[i][j] > target:
                j -= 1
            elif matrix[i][j] < target:
                i += 1
            else:
                return True
        return False
        

    

