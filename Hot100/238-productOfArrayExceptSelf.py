'''
Description: 除自身以外数组的乘积
Author: Luminary
Date: 2021-06-21 11:05:14
LastEditTime: 2021-06-21 11:05:35
'''
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 左右乘积列表：利用索引左侧所有数字乘积和右侧所有数字乘积（即前缀与后缀）相乘得到答案
        N = len(nums)
        left, right, res = [0] * N, [0] * N, [0] * N

        # 索引为0的元素左侧没有元素，因此初始化为1
        left[0] = 1
        for i in range(1, N):
            left[i] = left[i-1] * nums[i-1]
        
        # 索引为N-1的元素右侧没有元素，因此初始化为1
        right[N-1] = 1
        for i in range(N-2, -1, -1):
            right[i] = right[i+1] * nums[i+1]
        
        # 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
        for i in range(N):
            res[i] = left[i] * right[i]
        
        return res