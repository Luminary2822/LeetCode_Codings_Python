'''
Description: 使用最小花费爬楼梯
Author: Luminary
Date: 2021-05-15 21:46:55
LastEditTime: 2021-05-15 21:47:36
'''
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        N = len(cost)

        # d[i]表示爬到第i层需要的最低花费
        dp = [0] * N

        # 开始可以选择0或者1，在后面i=2的时候会判断选哪个更为最低
        dp[0] = cost[0]
        dp[1] = cost[1]

        for i in range(2, N):
            # dp[i]取决于之前爬一层上来还是爬两层上来最小值+当前花费
            dp[i] = min(dp[i-1],dp[i-2]) + cost[i]
        
        # N-1再踏一步就可以到楼顶，N-2再走两步就可以到楼顶，取最小花费
        return min(dp[N-1], dp[N-2])