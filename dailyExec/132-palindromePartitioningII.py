class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        # dp初始化为N，最大分割数字是每个字符都可以单独成一个回文
        # dp[i] 是以 i 结尾的分割成回文串的最少次数
        N = len(s)
        dp = [N] * N
        for i in range(N):
            # 0-i本身是一个回文串最小分割次数为0
            if self.isPalindrome(s[:i+1]):
                dp[i] = 0
                continue
            # 0-i不是回文串时候需要用 j 来切割 
            # 只有子字符串 s[j + 1..i] 是回文字符串的时候，dp[i] 可以通过 dp[j] 加上一个回文字符串 s[j + 1..i] 而得到
            for j in range(i):
                if self.isPalindrome(s[j+1:i+1]):
                    dp[i] = min(dp[i],dp[j]+1)
        return dp[-1]
    # 判断是否为回文串
    def isPalindrome(self, s):
        return s == s[::-1]

a = Solution()
print(a.minCut("aab"))

