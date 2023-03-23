'''
Description: 解码方法
Author: Luminary
Date: 2021-04-21 21:02:30
LastEditTime: 2021-04-21 21:03:00
'''
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # dp[i]表示前i个字符的解码方法数
        N = len(s)
        # 0位置空串，1-n对应s中0-n-1
        dp = [0] * (N+1)
        # 空串有一种解码方式
        dp[0] = 1
        for i in range(1, N+1):
            # 使用一个字符s[i-1]进行解码，s[i-1]不为'0'时，dp[i]取决于dp[i-1]，加入dp[i-1]解码方法数
            if s[i-1] != '0':
                dp[i] += dp[i-1]
            # 使用了两个字符即s[i-2]和s[i−1]进行解码，要满足s[i-2]不为‘0’且两个字符对应的数字在26之内
            # dp[i]取决于dp[i-2]，加入dp[i-2]解码方法数
            if i > 1 and s[i-2] != '0' and int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]
        return dp[N]