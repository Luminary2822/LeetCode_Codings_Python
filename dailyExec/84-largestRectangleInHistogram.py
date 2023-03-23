'''
Description: 柱状图中最大的矩形
Author: Luminary
Date: 2021-09-11 14:54:01
LastEditTime: 2021-09-11 14:54:01
'''
class Solution:
    def largestRectangleArea(self, heights):
        # 单调栈：从栈顶到栈底单调递减
        # 找每个柱子左右两边第一个小于该柱子的柱子
        heights = [0] + heights + [0]
        stack = []
        res = 0
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                mid = stack.pop()
                # 计算以第i根柱子为最矮柱子所能延伸的最大面积
                res = max(res, (i - stack[-1] - 1) * heights[mid])
            stack.append(i)
        return res