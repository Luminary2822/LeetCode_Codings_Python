'''
Description: 完全平方数
Author: Luminary
Date: 2021-04-30 14:45:25
LastEditTime: 2021-04-30 14:45:52
'''
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        # dp[i]表示组成i的完全平方数的最少个数
        dp = [0] * (n+1)
        for i in range(1, n+1):
            # 初始化组成i的完全平方数由i个1组成
            dp[i] = i
            # 遍历j从1到根号i，取dp[i]和减去j的完全平方数二者的最小值
            for j in range(1, int(math.sqrt(i)) + 1):
                    dp[i] = min(dp[i], dp[i-j*j] + 1)
        return dp[-1]