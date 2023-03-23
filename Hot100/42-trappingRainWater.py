'''
Description: 接雨水
Author: Luminary
Date: 2021-09-04 15:21:46
LastEditTime: 2021-09-04 15:21:48
'''
class Solution:
    def trap(self, height) :
        # 单调栈存储下标
        res = 0
        stack = []
        N = len(height)
        # 维护单调栈：从栈顶到栈底是从小到大顺序，当前h大于栈顶元素时出现凹槽，计算体积
        for i, h in enumerate(height):
            while stack and h > height[stack[-1]]:
                mid = stack.pop()
                # 注意如果出现弹出后栈内为空，则左边无柱子构不成凹槽直接将当前i入栈，将左侧更新为当前i的高度
                if not stack:
                    break
                left = stack[-1]

                currWidth = i - left - 1
                # 当前栈顶元素为中间凹槽底部的柱子下标，pop之后下一个栈顶就是其左边柱子下标，当前i为右边柱子下标
                currHeight = min(height[left],height[i]) - height[mid]
                # 计算当前位置体积累加到res中
                res += currWidth * currHeight
            stack.append(i)
        return res
