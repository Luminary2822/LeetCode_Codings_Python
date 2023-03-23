'''
Description: 雪糕的最大数量
Author: Luminary
Date: 2021-07-02 15:49:21
LastEditTime: 2021-07-02 17:30:34
'''
class Solution:
    def maxIceCream(self, costs, coins):
        '''
        一维背包超时，复杂度为 O(N* C)，N为物品数量，C为背包容量
        由于数据范围问题所以会超时
        '''
        # dp = [0] * (coins + 1)
        # for cost in costs:
        #     for i in range(coins, cost - 1, -1):
        #         dp[i] = max(dp[i], dp[i-cost] + 1)
        # return dp[coins]

        # 采用贪心算法
        # 优先选择价格小的物品会使得我们剩余金额尽可能的多，将来能够做的决策方案也就相应变多
        # 对物品数组进行「从小到大」排序，然后「从前往后」开始决策购买
        costs.sort()
        for i in range(len(costs)):
            # 如果钱花完但是雪糕还有，就直接返回i，因为循环已经进入到买不起的那个了
            if coins < costs[i]:
               return i
            coins -= costs[i]
        # 如果都买得起说明costs数组遍历完，返回数组长度i+1
        return i + 1
        