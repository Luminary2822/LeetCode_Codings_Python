class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        # dp[i] 表示以 i 结尾的最长递增子序列的长度
        if not envelopes:
            return 0
        N = len(envelopes)
        # 按照先第一维升序，一维相同按第二维降序排列
        envelopes.sort(key=lambda x:(x[0], -x[1]))
        # 存储以 i 结尾最长递增子序列的长度，初始化只有一个信封
        dp = [1] * N
        for i in range(N):
            for j in range(i):
                # 比较第二维度，满足情况更新状态转移方程
                if envelopes[j][1] < envelopes [i][1]:
                    # 第i个信封是必须选择的，前i-1的信封有符合条件的就要更新dp[i]
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
