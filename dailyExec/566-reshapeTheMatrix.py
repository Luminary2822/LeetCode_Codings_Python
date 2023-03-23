# 重塑数组
class Solution(object):
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        # 获取原数组行列数
        M,N = len(nums), len(nums[0])
        # 无法重塑的情况下返回原数组
        if M * N != r * c:
            return nums
        row,col = 0,0
        # 新建r行c列值为0的结果数组
        res = [[0]*c for _ in range(r)]
        for i in range(M):
            for j in range(N):
                # 到达最后一列，新起一行
                if col == c:
                    row += 1
                    col = 0
                # 逐行遍历放置到对应位置
                res[row][col] = nums[i][j]
                col += 1
        return res
        # numpy的reshape方法
        """
        import numpy
        return np.asarray(nums).reshape((r, c))
        """        