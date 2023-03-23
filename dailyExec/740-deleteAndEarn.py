'''
Description: 删除并获得点数
Author: Luminary
Date: 2021-05-10 16:33:41
LastEditTime: 2021-05-10 16:34:07
'''
class Solution(object):
    def deleteAndEarn(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 将数组转换成每家的点数，转换成打家劫舍问题，不能取相邻的
        # [3,4,2]转换成[0,0,2,3,4]
        # [2,2,3,3,3,4]转换成[0,0,4,9,4]

        # 转换数组的长度取决于原数组中的最大值
        trans_len = max(nums) + 1
        trans = [0] * trans_len

        # 累加每家的点数
        for i in range(len(nums)):
            trans[nums[i]] += nums[i]
        
        # dp[i]表示经过操作可以获得的最大点数
        dp = [0] * trans_len
        dp[0] = trans[0]
        dp[1] = max(trans[0], trans[1])

        for i in range(2, trans_len):
            dp[i] = max(trans[i] + dp[i-2], dp[i-1])
    
        return dp[-1]