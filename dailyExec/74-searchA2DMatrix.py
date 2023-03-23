class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        M = len(matrix)
        N = len(matrix[0])
        for i in range(M):
            # 升序性质，将目标元素与每行最后一个元素比较
            if target > matrix[i][N-1]:
                continue
            # 如果目标小于当前行的最大元素，由于下面的行内元素均是大于当前行，所以目标如果存在的话就在当前行内
            if target in matrix[i]:
                return True
        return False
        # 全局二分法：二维矩阵当做一维矩阵，前提每一行最后一个元素小于下一行第一个元素
        """
        M,N = len(matrix),len(matrix[0])
        left, right = 0, N-1
        while left <= right:
            # 根据 mid 求出在二维矩阵中的具体位置
            mid = left + (right-left) //2
            cur = matrix[mid // N][mid % N]
            # 判断 left 和 right 的移动方式
            if cur == target:
                return True
            elif cur < target:
                left = mid + 1
            else:
                right = mid - 1
        return False
        """
        