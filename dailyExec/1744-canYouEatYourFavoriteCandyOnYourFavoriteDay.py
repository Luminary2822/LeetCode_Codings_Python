'''
Description: 你能在你最喜欢的那天吃到你最喜欢的糖果吗？能！
Author: Luminary
Date: 2021-06-01 11:20:37
LastEditTime: 2021-06-01 11:21:07
'''
class Solution(object):
    def canEat(self, candiesCount, queries):
        """
        :type candiesCount: List[int]
        :type queries: List[List[int]]
        :rtype: List[bool]
        """
        
        n = len(candiesCount)
        # candiesCount前缀和:得到想把前i种水果吃完需要吃多少个，前i-1类糖果总和
        preSum = [0] * (n+1)
        for i in range(n):
            preSum[i+1] = preSum[i] + candiesCount[i]
        
        m = len(queries)
        ans = [False] * m
        for i, (t,d,limit) in enumerate(queries):
            # 到favoriteDayi可以吃到的最小糖果数
            min_candy = d + 1
            # 到favoriteDayi可以吃到的最大糖果数
            max_candy = limit * (d + 1)
            # 尽量少吃不能吃到下一种，尽量多吃不能小于前一种
            if min_candy <= preSum[t+1] and max_candy > preSum[t]:
                ans[i] = True
        return ans