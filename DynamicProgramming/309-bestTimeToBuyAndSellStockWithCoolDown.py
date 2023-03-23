'''
Description: 最佳买卖股票时机含冷冻期
Author: Luminary
Date: 2021-07-05 14:35:04
LastEditTime: 2021-07-05 17:05:29
'''
class Solution:
    def maxProfit(self, prices):
        # dp[i][0]表示第i天买入状态所持最大金额
        # dp[i][1]表示第i天卖出状态所持最大金额
        if len(prices) == 1:return 0
        dp = [[0, 0] for _ in range(len(prices))]
        # 第0天买入
        dp[0][0] = -prices[0]
        # 第一天买入 = max(维持昨天买入，前天卖出0 - 今天买入)
        dp[1][0] = max(-prices[0], -prices[1])
        # 第一天卖出 = max(维持昨天卖出，昨天买入今天卖出)
        dp[1][1] = max(dp[0][1], dp[0][0] + prices[1])
        for i in range(2, len(prices)):
            # 第i天买入 = max(维持昨天买入，前天卖出 - 今天买入（含冷冻期）)
            dp[i][0] = max(dp[i-1][0], dp[i-2][1] - prices[i])
            # 第i天卖出 = max(维持昨天卖出，昨天买入 + 今天卖出)
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        return dp[-1][-1]