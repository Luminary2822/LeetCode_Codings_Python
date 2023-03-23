# 最大子序和-动态规划
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 动态规划
        # 状态定义：dp[i]代表以nums[i]结尾的最大子序列和
        # 转移方程：dp[i]等于dp[i-1]+nums[i]、nums[i]二者最大值
        dp = [nums[0]]
        for i in range(1, len(nums)):
            dp.append(max((dp[i-1]+nums[i]), nums[i]))
        return max(dp)
        # 简便写法,nums[i]更新为存储到目前为止最大子序列和
        # nums[i-1]为前一项最大子序列和，与0比较
        # 大于0则加上当前项赋予到nums[i]表示目前的最大子序列和，小于0则nums[i] = 自身
        """
        for i in range(1, len(nums)):
            # 等式左边nums[i]表示以其为结尾的最大子序列和
            # 右边第一个nums[i]为nums数组本身原值，nums[i-1]和第一个nums[i]内涵相同
            nums[i]= nums[i] + max(nums[i-1], 0)
        return max(nums)
        """
        