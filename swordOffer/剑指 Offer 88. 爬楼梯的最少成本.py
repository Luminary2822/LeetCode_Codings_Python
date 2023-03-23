'''
Description: 爬楼梯的最少成本
Author: Luminary
Date: 2021-09-16 19:33:54
LastEditTime: 2021-09-16 19:34:08
'''
class Solution:
    def minCostClimbingStairs(self, cost) :
        # dp[i]表示爬到第i层需要花费的体力值
        N = len(cost)
        dp = [0] * N
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2, len(cost)):
            dp[i] = min(dp[i-1],dp[i-2]) + cost[i]
        return min(dp[N-1],dp[N-2])
