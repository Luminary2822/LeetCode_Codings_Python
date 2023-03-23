'''
Description: 不同路径，左上角到右下角的路径数量
Author: Luminary
Date: 2021-04-20 10:44:50
LastEditTime: 2021-04-20 10:45:31
'''
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # 这题类似于64-最小路径和，采用动态规划
        # dp[i][j]表示从左上角到达右下角的路径总数
        dp = [[0] * n for _ in range(m)]
        # 第一列和第一行由于只有单方向所以路径数量一直为1
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        # 其他位置的路径数量等于来自两个方向的数量之和
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]