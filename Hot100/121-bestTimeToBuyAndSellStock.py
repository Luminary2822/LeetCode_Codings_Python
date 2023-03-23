'''
Description: 买卖股票的最佳时机
Author: Luminary
Date: 2021-04-22 11:02:52
LastEditTime: 2021-04-22 11:03:15
'''
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        N = len(prices)
        # dp[i]表示前 i天获得的最大收益
        dp = [0] * N
        # 记录目前遍历过的最小值
        min_price = prices[0]
        for i in range(1, N):
            # 前i天获得的最大收益等于：max(前i-1天获得的最大收益与当天卖出可获得最大收益)
            dp[i] = max(dp[i-1], prices[i] - min_price)
            # 记录前i-1天的最小值
            min_price = min(prices[i], min_price)
        return dp[N-1]