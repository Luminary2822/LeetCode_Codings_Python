# 螺旋向内遍历矩阵，遍历的过程中依次放入 1-N^2的各个数字
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        if n == 0: return []
        # 初始化n*n结果矩阵
        res = [[0] * n for i in range(n)]
        # 四个方向的边界
        left, right, up, down = 0, n - 1, 0, n - 1
        # 当前位置
        x, y = 0, 0
        # 移动方向：右下左上，移动到了边界更改方向
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # 表示当前的移动方向的下标，dirs[cur_d] 就是下一个方向需要怎么修改 x, y
        cur_d = 0
        count = 0
        # 结束条件count元素个数等于n^2
        while count != n * n:
            # 矩阵按照遍历到的位置赋值
            res[x][y] = count + 1
            count += 1
            # 到达右边界，cur_d更改移动方向由右到下，当前行遍历结束，up边界向内移动一格
            if cur_d == 0 and y == right:
                cur_d += 1
                up += 1
            # 到达下边界，cur_d更改移动方向由下到左，当前列遍历结束，right边界想内移动一格
            elif cur_d == 1 and x == down:
                cur_d += 1
                right -= 1
            # 同理
            elif cur_d == 2 and y == left:
                cur_d += 1
                down -= 1
            elif cur_d == 3 and x == up:
                cur_d += 1
                left += 1
            cur_d %= 4
            # 获取矩阵位置坐标
            x += dirs[cur_d][0]
            y += dirs[cur_d][1]
        return res