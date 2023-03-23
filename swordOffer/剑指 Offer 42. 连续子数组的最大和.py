'''
Description: 连续子数组的最大和
Author: Luminary
Date: 2021-07-17 11:56:58
LastEditTime: 2021-07-17 12:08:58
'''
class Solution:
    def maxSubArray(self, nums):
        # dp[i]表示以nums[i]结尾的最大子数组和
        N = len(nums)
        dp =[0 for _ in range(N)]
        dp[0] = nums[0]
        for i in range(1, N):
            dp[i] = max(dp[i-1] + nums[i], nums[i])
        return max(dp)
        # # 前缀和
        # res = 0
        # ans = float('-inf')
        # # 计算前缀和，实时更新最值，当前缀和小于0时重新计算
        # for i in range(len(nums)):
        #     res += nums[i]
        #     ans = max(res, ans)
        #     if res < 0:res = 0
        # return ans
