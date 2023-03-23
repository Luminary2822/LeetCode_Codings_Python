'''
Description: 矩阵区域不超过K的最大数值和
Author: Luminary
Date: 2021-04-23 11:06:41
LastEditTime: 2021-04-23 11:20:45
'''
import bisect
class Solution(object):
    def maxSumSubmatrix(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        # 记录行和列
        row , col = len(matrix) , len(matrix[0])
        # 记录矩阵和
        res = float("-inf")
        # left和right索引表示列的范围
        for left in range(col):
            # 列累加和列表
            sums = [0] * row
            for right in range(left,col):
                for j in range(row):
                    # 同一行的左右列逐渐相加，转换为在一维数组sums中判断最大矩形和
                    sums[j] += matrix[j][right]
                # 存放对于sums中元素累加的和，lst[i]表示在sums中i位置之前的累加和
                lst = [0]
                # 用来累加之前算出来的累加列表
                cur = 0
                for num in sums:
                    cur += num
                    # 寻找cur-k是否存在用来判断sums(0,j)-sums(0,i-1)<=k是否满足
                    loc = bisect.bisect_left(lst,cur-k)
                    if loc < len(lst):
                        # 存在记录当前区域的数值和
                        res = max(cur-lst[loc],res)
                    # 插入元素保持原有顺序
                    bisect.insort(lst,cur)
        return res