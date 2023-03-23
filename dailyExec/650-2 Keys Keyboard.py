'''
Description: 只有两个键的键盘
Author: Luminary
Date: 2021-09-19 15:30:29
LastEditTime: 2021-09-19 15:30:30
'''
class Solution:
    def minSteps(self, n):
        # 动态规划：n是质数结果还是n只能一个一个复制，n是合数结果是因数所需最小操作次数+复制粘贴因数次数

        if n == 1:return 0
        # dp[i]表示打印i个‘A’的最少操作次数
        dp = [0] * (n+1)

        for i in range(2, n+1):
            dp[i] = i
            # i是合数的情况下寻找i的最大因数更新dp[i]
            for j in range(int(i/2), 1, -1):
                # 如果j是i的因数，i个可以由j个复制粘贴得到，复制粘贴次数为i/j
                if i % j == 0:
                    dp[i] = dp[j] + int(i/j)
                    break # 因为因数之前计算过，可以保证所需的步数最少，所以j从大到小可以break
        return dp[n]