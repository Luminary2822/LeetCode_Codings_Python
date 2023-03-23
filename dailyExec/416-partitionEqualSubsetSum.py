'''
Description: 分割等和子集
Author: Luminary
Date: 2021-07-27 21:55:12
LastEditTime: 2021-07-27 21:55:54
'''
class Solution:
    def canPartition(self, nums):
        # 背包容量为sum/2, 看组成sum/2背包的方法是否存在

        # 如果数字之和不是2的倍数，那必然不可能分割成功（因为整数相加不可能等于小数）
        sum_nums = sum(nums)
        if sum_nums % 2 != 0: 
            return False
        # 目标背包容量为target，判断能否组成该背包
        target = sum_nums // 2

        # dp[i]表示装满容量为i的背包有多少种方法，容量为0是一种方法即什么都不选
        dp = [0] * (target + 1)
        dp[0] = 1
        for num in nums:
            for i in range(target, num-1, -1):
                dp[i] += dp[i-num]
        # d[target]方法数是否为0，为0说明false，不为0说明true
        return dp[-1] != 0