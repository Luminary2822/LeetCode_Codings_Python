class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        result = []
        # [0,rowIndex]
        for i in range(rowIndex+1):
            # 每行生成对应个数全为1的列表
            now = [1] * (i+1)
            for n in range(1, i):
                # 当前位置等于前一行当前位置-1元素值+前一行当前位置值
                now[n] = pre[n-1] + pre[n]
            result = now
            pre = now
        return result

        # 动态规划滚动数组：优化空间复杂度
        # 使用一维数组，然后从右向左遍历每个位置，每个位置的元素 res[j] += 其左边的元素 res[j−1]。
        """
        dp = [1] * (rowIndex + 1)
        for i in range(2, rowIndex + 1):
            for j in range(i-1, 0, -1):
                dp[j] = dp[j] + dp[j-1]
        return dp
        """

a = Solution()
print(a.getRow(1))