'''
Description: 目标和（0-1背包）
Author: Luminary
Date: 2021-06-19 16:26:44
LastEditTime: 2021-06-19 16:29:28
'''
class Solution(object):
    def findTargetSumWays(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # dp[i]表示能组合成容量为i的方式有多少种
        # sum(+) - sum(-) = target
        # sum_num + sum(+) - sum(-) = target + sum_num
        # 2 sum(+) = target + sum_num
        # sum(+) = (target + sum_num) // 2

        # 题意分析
        sum_num = sum(nums)
        new_bag = (target + sum_num) // 2
        if sum_num < target or (sum_num + target) % 2 == 1:
            return 0
        
        # 目标组成容量为new_bag的背包方式有多少
        dp = [0] * (new_bag + 1)
        # 当背包容量为 0 时，只有一种方法能够满足条件，就是什么也不选
        dp[0] = 1

        # 逆序遍历：当前new_bag容量到num容量依次计算方法数
        for num in nums:
            for i in range(new_bag, num-1, -1):
                dp[i] += dp[i-num]
        return dp[-1]
    
a = Solution()
print(a.findTargetSumWays([1,1,1,1,1],3))