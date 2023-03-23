'''
Description: 两个字符串的删除操作
Author: Luminary
Date: 2021-09-25 21:11:53
LastEditTime: 2021-09-25 21:13:12
'''
class Solution:
    def minDistance(self, word1, word2):
        # 寻找两个字符串的最长公共子序列
        # 结尾：用两个字符串长度之和减去2倍的最长公共子序列就是需要的步数
        M = len(word1)
        N = len(word2)
        dp = [[0] * (N + 1) for _ in range(M+1)]
        for i in range(1, M+1):
            for j in range(1, N+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return M + N - 2 * dp[M][N]
