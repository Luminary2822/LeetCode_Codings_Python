'''
Description: 最长回文子序列
Author: Luminary
Date: 2021-09-01 20:41:54
LastEditTime: 2021-09-01 20:42:07
'''
class Solution:
    def longestPalindromeSubseq(self, s):
        # dp[i][j]表示s[i:j]最长回文子序列的长度
        N = len(s)
        dp = [[0] * N for _ in range(N)]
        # 每个字符都是单独的回文子序列
        for i in range(N):
            dp[i][i] = 1
        for i in range(N-1, -1, -1):
            for j in range(i+1, N):
                # 判断s[i]与s[j]是否相同，相同则长度加2，不同则判断加入其中之一哪个形成的回文子序列最长
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return dp[0][-1]