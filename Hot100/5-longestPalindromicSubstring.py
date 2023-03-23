class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        N = len(s)
        # dp[i][j]表示s[i:j]是否为回文串
        dp = [[False] * N for _ in range(N)]
        start,maxLen = 0,1
        # 边界判断
        if N < 2:
            return s
        # 初始化：每个字母均为独立的回文串
        for i in range(N):
            dp[i][i] = True
        # 枚举终点和起点
        for j in range(1,N):
            for i in range(0,j):
                if s[i] == s[j]:
                    # 小于3则设为回文串
                    if j - i < 3:
                        dp[i][j] = True
                    # 否则取决于子串是否为回文串
                    else:
                        dp[i][j] = dp[i+1][j-1]
                # 更新完dp[i][j]之后，更新回文串max_len和起始位置start
                if dp[i][j]:
                    tempLen = j-i+1
                    if tempLen > maxLen:
                        maxLen = tempLen
                        start = i
        return s[start: start+maxLen]
                

