'''
Description: 最长递增子序列的个数
Author: Luminary
Date: 2021-09-20 20:40:49
LastEditTime: 2021-09-20 20:40:50
'''
class Solution:
    def findNumberOfLIS(self, nums) :
        # dp[i]以nums[i]结尾最长递增子序列的长度
        # count[i]以num[i]为结尾的最长递增子序列个数
        N = len(nums)
        if N <= 1: return N
        dp = [1 for i in range(N)]
        count = [1 for i in range(N)]
        
        maxCount = 0    # 记录最长递增子序列的长度
        for i in range(N):
            for j in range(i):
                if nums[i] > nums[j]:
                    # 找到一个更长的递增子序列
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    # 找到了两个相同长度的递增子序列
                    elif dp[j] + 1 == dp[i]:
                        count[i] += count[j]
                maxCount = max(maxCount,dp[i])
        
        # 再遍历一遍，把最长递增序列长度对应的count[i]累计下来
        result = 0
        for i in range(N):
            if dp[i] == maxCount:
                result += count[i]
        return result
                