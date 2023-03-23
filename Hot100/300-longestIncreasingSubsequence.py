class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        N = len(nums)
        # dp[i]以第 i个数字结尾的最长上升子序列的长度，初始化为 1
        dp = [1] * N
        for i in range(N):
            for j in range(i):
                # nums[i]必须被选取，当nums[i] > nums[j]，表明dp[i]可以从dp[j]的状态转移过来
                # nums[i] 可以放在nums[j] 后面以形成更长的上升子序列
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        # 最后返回 dp 数组中最大值
        return max(dp)