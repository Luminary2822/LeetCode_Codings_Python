'''
Description: 岛屿数量
Author: Luminary
Date: 2021-09-01 20:53:27
LastEditTime: 2021-09-01 20:54:16
'''
class Solution:
    # 深度优先搜索
    # 以位置1为起点进行深度优先搜索，搜索到的1标记为0
    def dfs(self, grid, r, c):
        grid[r][c] = 0
        nr, nc = len(grid), len(grid[0])
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs(grid, x, y)

    # 最终岛屿的数量就是深度优先搜索的次数
    def numIslands(self, grid) :
        nr, nc = len(grid), len(grid[0])
        if nr == 0:
            return 0
        num_island = 0
        for i in range(nr):
            for j in range(nc):
                if grid[i][j] == "1":
                    num_island += 1
                    self.dfs(grid, i, j)
        return num_island