'''
Description: 单词拆分
Author: Luminary
Date: 2021-07-16 13:42:41
LastEditTime: 2021-07-16 13:42:58
'''
class Solution:
    def wordBreak(self, s, wordDict) :
        # dp[i]表示以i-1结尾的字符串是否可以被wordDict查分
        N = len(s)
        dp = [False for _ in range(N + 1)]
        dp[0] = True
        
        for i in range(1, N + 1):
            for j in range(i):
                # dp[i]为true的前提是：dp[j]为true且j到i之间的单词在单词表中
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    # 一旦dp[i]为True了，后面的j也不用遍历
                    break
        return dp[-1]