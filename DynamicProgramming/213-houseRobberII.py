'''
Description: 打家截舍II（首尾可相连数组）
Author: Luminary
Date: 2021-04-15 17:03:04
LastEditTime: 2021-04-15 17:18:03
'''
class Solution(object):
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
        # 将环拆成两个队列，[0, N-1]和[1, N]
        return max(self.rob1(nums[0:N-1]), self.rob1(nums[1:N]))

    # 非环数组情况，打家劫舍I
    def rob1(self, nums):
        N = len(nums)
        if N == 0:
            return 0
        if N == 1:
            return nums[0]
        # dp[i]表示满足不触动警报条件前i家可以偷窃到的最高金额
        dp = [0] * N
        # 只有一家时只能选择该家
        dp[0] = nums[0]
        # 有两家时要选一家金额最多的
        dp[1] = max(nums[0],nums[1])
        for i in range(2, N):
            # 是否选择当前i位置家进行打劫，如果选择则在i-2基础上加，如果不选即为dp[i-1]
            dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        return dp[N-1]