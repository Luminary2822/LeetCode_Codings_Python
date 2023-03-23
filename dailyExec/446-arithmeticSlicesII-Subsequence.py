'''
Description: 等差数列划分 II - 子序列
Author: Luminary
Date: 2021-09-01 20:17:12
LastEditTime: 2021-09-01 20:21:06
'''
from collections import defaultdict
class Solution:
    def numberOfArithmeticSlices(self, nums) :
        # dp[i][j] 代表以 nums[i] 为结尾数字，能够组成公差为 j 的等差数列的个数。
        # 因为以nums[i]结尾的等差数列可能存在多个公差的等差数列
        # 所以第二维度用哈希表存储不同公差对应等差数列的个数
        N = len(nums)
        dp = [defaultdict(int) for _ in range(N+1)]
        res = 0
        for i in range(N):
            for j in range(i):
                d = nums[i] - nums[j]
                # 更新 i 下面的 hashmap
                dp[i][d] += dp[j][d] + 1
                # 前一个结果作为数组下标 i 可以组成的等差数列数组个数
                res += dp[j][d]
        return res