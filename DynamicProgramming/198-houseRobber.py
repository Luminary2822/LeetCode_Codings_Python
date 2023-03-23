'''
Description: 打家截舍I
Author: Luminary
Date: 2021-04-15 17:03:04
LastEditTime: 2021-04-15 17:16:46
'''
class Solution:
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        N = len(nums)
        if N == 0:
            return 0
        if N == 1:
            return nums[0]
        # 状态定义：dp[i]为满足不触动警报条件下前i家偷窃到的累计最高金额
        dp = [0] * N
        dp[0] = nums[0]
        dp[1] = max(nums[0],nums[1])
        # 当前是否选择nums[i]，比较选择或者不选的最值定义状态转移方程
        for i in range(2, N):
            dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        return dp[-1]
