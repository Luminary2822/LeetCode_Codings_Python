'''
Description: 斐波那契数列
Author: Luminary
Date: 2021-10-05 14:16:04
LastEditTime: 2021-10-05 14:17:04
'''
class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = (dp[i-1] + dp[i-2]) % 1000000007
        return dp[n]