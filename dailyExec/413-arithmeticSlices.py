'''
Description: 等差数列划分
Author: Luminary
Date: 2021-09-01 20:15:41
LastEditTime: 2021-09-01 20:16:03
'''
class Solution:
    def numberOfArithmeticSlices(self, nums):
        # dp[i]表示以nums[i]结尾的等差数列数组的个数
        N = len(nums)
        dp = [0] * N
        for i in range(2, N):
            # 新增加的nums[i]可以和前面构成等差数列，dp[i] = dp[i-1] + 1
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                dp[i] = dp[i-1] + 1
        # 数组中的等差数列的数目
        return sum(dp)