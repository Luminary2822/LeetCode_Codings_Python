'''
Description: 不同路径II
Author: Luminary
Date: 2021-09-23 13:34:03
LastEditTime: 2021-09-23 13:34:03
'''
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid):
        # 与不同路径I区别的是：处理有障碍的地方dp应为0，第一行和第一列有障碍的位置后面全是0   

        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        # dp数组表示从(0,0)出发到(i,j)有dp[i][j]条不同的路径。
        dp = [[0] * n for _ in range(m)]

        # 初始位置为障碍则直接返回0条路径
        if obstacleGrid[0][0] == 1:return 0

        # 第一列和第一行，没有障碍更新dp，有障碍则跳出循环后面dp数组均为0
        for i in range(m):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = 1
            else:
                break
        for i in range(n):
            if obstacleGrid[0][i] == 0:
                dp[0][i] = 1
            else:
                break
        
        # 没有障碍更新dp来自两个方向，有障碍则继续循环该位置的dp为0
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:continue
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
        
            