'''
Description: 扰乱字符串
Author: Luminary
Date: 2021-04-16 15:10:58
LastEditTime: 2021-04-16 19:42:11
'''
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        # 空判
        if not grid or not grid[0]:
            return 0
        
        # dp[i][j]表示从左上角到(i,j)位置的最短路径和
        M, N= len(grid), len(grid[0])
        dp = [[0] * N for _ in range(M)]

        # 左上角元素单独定义
        dp[0][0] = grid[0][0]

        # 对于第一行元素，路径只能来自于左边元素向右走加上当前位置
        for j in range(1, N):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        # 对于第一列元素，路径只能来自于上边元素向下走加上当前位置
        for i in range(1, M):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        # 对于其他元素，路径来自于上面或者左边路径的最小值加上当前元素
        for i in range(1, M):
            for j in range(1, N):
                dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i][j]
        return dp[M-1][N-1]

