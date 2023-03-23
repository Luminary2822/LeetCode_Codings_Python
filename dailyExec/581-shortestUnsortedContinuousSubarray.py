'''
Description: 最短无序连续子数组
Author: Luminary
Date: 2021-07-16 15:29:48
LastEditTime: 2021-07-16 15:30:48
'''
class Solution:
    def findUnsortedSubarray(self, nums) :
        # 正序遍历寻找右端点
        max_num = nums[0]
        right = 0
        for i in range(0, len(nums)):
            # 说明当前num[i]小于前面的最大值为错误的元素，nums[i]应该包括到结果区间，右端点应该是i
            if nums[i] < max_num:
                right = i
            max_num = max(nums[i], max_num)

        # 逆序遍历寻找左端点
        min_num = nums[-1]
        left = len(nums) 
        for i in range(len(nums) - 1, -1, -1):
            # 说明当前num[i]大于后面的最小值为错误的元素，nums[i]应该包括到结果区间，左端点应该是i
            if nums[i] > min_num:
                left = i
            min_num = min(nums[i], min_num)
            
        return max(right - left + 1, 0)