'''
Description: 组合总数IV
Author: Luminary
Date: 2021-04-25 16:51:43
LastEditTime: 2021-04-25 17:05:09
'''
class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # dp[i] 表示从 nums 中挑选数字可以构成 i 的方法数
        dp = [0] * (target + 1)
        # dp[0] 表示从数组中抽取任何元素组合成 0 的方案数
        # 在递归的时候 target == 0，说明在 for 循环中的 target - num 得到了 0，
        # 表示 nums 数组中恰好有一个数字等于 target。所以返回 1，因此需要令 dp[0] = 1
        dp[0] = 1
        # 求dp[1-target]总和为1-target的组合个数
        for i in range(target + 1):
            # 遍历nums
            for num in nums:
                # 判断当构成target选择nums[i]的情况下
                if i >= num:
                    # 要求有多少方法可以构成target-nums[i]，即求dp[target-nums[i]]，递归过程
                    # target为4，在选择nums[0]=1的情况下，要求有多少种方法可以构成4-1=3，累加dp[3]
                    dp[i] += dp[i - num]
        return dp[target]