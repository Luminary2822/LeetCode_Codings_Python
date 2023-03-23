'''
Description: 整数拆分
Author: Luminary
Date: 2021-09-25 21:42:11
LastEditTime: 2021-09-25 21:42:12
'''
class Solution:
    def integerBreak(self, n):
        # dp[i]分拆数字可以得到的最大乘积
        dp = [0] * (n + 1)
        # 0和1无解，所以只初始化dp[2]
        dp[2] = 1
        # i从3开始，j从1开始，dp[i-j]就是dp[2]可以通过初始化的值来求
        for i in range(3, n+1):
            for j in range(1, i):
                # 两种渠道得到dp[i]，拆分成j和i-j两个数相乘，j*dp[i-j]拆分成多个数相乘
                dp[i] = max(dp[i], j * (i-j), j * dp[i-j])
        return dp[n]