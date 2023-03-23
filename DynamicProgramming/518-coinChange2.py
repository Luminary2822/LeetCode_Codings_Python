'''
Description: 零钱兑换II
Author: Luminary
Date: 2021-06-10 19:52:29
LastEditTime: 2021-06-10 19:52:46
'''
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        # dp[i] 表示金额之和等于 i 的硬币组合数
        dp = [0] * (amount + 1)
        # 不选取任何硬币时，组合金额为0，即有一种组合方法dp[0]=1
        dp[0] = 1

        # 遍历对于每一种面额的硬币，更新数组dp中的每个大于或等于该面额的元素的值。
        for coin in coins:
            for i in range(coin, amount + 1):
                # i的组合数 = 当前coin结合i-coin的组合数 
                dp[i] += dp[i - coin]
        return dp[amount]