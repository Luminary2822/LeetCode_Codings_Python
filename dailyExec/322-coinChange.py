'''
Description: 零钱兑换
Author: Luminary
Date: 2021-06-11 22:20:08
LastEditTime: 2021-06-11 22:20:40
'''
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # dp[i]表示凑成金额i所需的最少硬币个数,初始化最大值
        dp = [float('inf')] * (amount + 1)
        # 组成0金额需要0个硬币
        dp[0] = 0
        # 对于每一种金额所需硬币个数取min(【当前值】和【i减去coin所需硬币个数加上当前硬币】)
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i-coin] + 1)
        
        # 如果没有任何一种硬币组合能组成总金额，返回 -1
        if dp[amount] == float('inf'):
            return -1
        else:
            return dp[amount]