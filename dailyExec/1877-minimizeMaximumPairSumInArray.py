'''
Description: 数组中最大数对和的最小值
Author: Luminary
Date: 2021-07-20 16:57:17
LastEditTime: 2021-07-20 16:59:38
'''
class Solution:
    def minPairSum(self, nums):
        # 贪心
        # 尽量让较小数和较大数组成数对
        # 对原数组 nums 进行排序，然后从一头一尾开始往中间组「数对」，取所有数对中的最大值
        nums.sort()
        left= 0
        right = len(nums) - 1
        res = 0
        for i in range(len(nums)):
            if left > right: break
            res = max(res, nums[left] + nums[right])
            left += 1
            right -= 1
        return res
