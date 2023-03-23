'''
Description: 一和零
Author: Luminary
Date: 2021-06-06 20:42:53
LastEditTime: 2021-06-06 20:43:30
'''
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        # 二维dp，背包问题

        # dp[i][j]表示用i个0和j个1可以组成最大元素的个数
        dp = [[0]* (n+1) for _ in range(m+1)]

        for s in strs:
            # 对于每一个0,1串，统计一下0和1的个数
            zero_num = s.count('0')
            one_num = len(s) - zero_num

            # 对于可容纳的背包要依次遍历，在（不放）和（放+当前）两种背包容量选择较大值
            for i in range(m, zero_num - 1, -1):
                for j in range(n, one_num - 1, -1):
                    dp[i][j] = max(dp[i][j], 1 + dp[i - zero_num][j - one_num])
        
        return dp[m][n]