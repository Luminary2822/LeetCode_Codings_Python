'''
Description: 爬楼梯（一维动态规划）
Author: Luminary
Date: 2021-04-16 11:14:52
LastEditTime: 2021-04-16 11:33:01
'''
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        # dp[i]表示爬到第i层楼梯有的方法数
        # 初始化成n+1是将n=2的情况包括进迭代中，以免特殊情况判断
        dp = [0] * (n+1)
        dp[0] = dp[1] = 1
        for i in range(2,n+1):
            # 第n个台阶只能从第n-1或者n-2个上来。
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]
        
        # # 这是初始化n，特殊判断前两层
        # if n == 1 or n == 2:return n
        # dp = [0] * (n)
        # dp[0] = dp[1] = 1
        # # n >= 3时才会进入迭代计算
        # for i in range(2,n):
        #     dp[i] = dp[i-1] + dp[i-2]
        # return dp[n-1]