'''
Description: 寻找峰值
Author: Luminary
Date: 2021-09-15 11:16:47
LastEditTime: 2021-09-15 11:32:20
'''
class Solution:
    def findPeakElement(self, nums) :
        # 二分查找
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            # 峰值可能出现在mid右侧
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            # 峰值可能出现在mid以及mid向左处
            else:
                right = mid
        # 当left和right相等指向峰值的时候，返回left
        return left
        
        # # 自己写的方法：构建栈顶到栈底单调递减的单调栈
        # stack = []
        # for i in range(len(nums)):
        #     # 当前元素小于栈顶元素时，栈顶元素即为峰值
        #     if stack and nums[i] < nums[stack[-1]]:
        #         return stack[-1]
        #     # 当前元素大于等于栈顶元素时，将元素入栈
        #     else:
        #         stack.append(i)
        # # 如果没有在循环内部返回可能是单调序列，直接返回单调递减栈的栈顶元素即为峰值
        # return stack[-1]
    