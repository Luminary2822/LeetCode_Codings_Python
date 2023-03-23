class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        """
        :type text1: str
        :type text2: str
        :rtype: int
        """
        # dp[i][j] 表示 text1[0:i-1](包括 i - 1) 和 text2[0:j-1](包括 j - 1) 的最长公共子序列
        M, N = len(text1), len(text2)
        # 便于当 i = 0 或者 j = 0 的时候，dp[i][j]表示的为空字符串和另外一个字符串的匹配
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        # dp[i][j] = 0，无需再遍历
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                # 两个子字符串的最后一位相等，所以最长公共子序列又增加了 1
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    # 两个子字符串的最后一位不相等，此时目前最长公共子序列长度为两个字符串其中之一去除当前不相等字符前面子串与另一字符串的匹配
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[M][N]
