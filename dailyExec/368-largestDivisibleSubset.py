'''
Description: 最大整除子集
Author: Luminary
Date: 2021-04-23 11:25:53
LastEditTime: 2021-04-23 11:26:08
'''
class Solution(object):
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # dp[i]表示i位置对应元素的最长整除子集（里面所有元素均为nums[i]的因数）
        N = len(nums)
        if not nums: return nums
        if N == 1: return nums

        # 排序过后只存在nums[i] % nums[j] == 0的情况
        nums.sort()

        # 所有位置子集初始化为自身元素，自己是自己的因数
        dp = [[num] for num in nums]

        # 寻找i位置前所有nums[i]的因数加入dp[i]
        for i in range(1, N):
            # for j in range(i-1, -1, -1):
            for j in range(0, i):
                if nums[i] % nums[j] == 0:
                    # 如果nums[j]是nums[i]的因数，那么dp[j]全部均为nums[i]的因数
                    # 比较当前因数子集的长度与由dp[j]加入nums[i]后因数子集的长度，选取最长者
                    dp[i] = max(dp[j] + [nums[i]], dp[i], key=len)
                    
        # 最后返回所有位置中最大整除子集, 比较关键条件为长度
        return max(dp, key=len)