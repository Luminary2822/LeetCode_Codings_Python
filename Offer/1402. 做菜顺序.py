'''
Description: 做菜顺序
Author: Luminary
Date: 2021-09-02 21:22:55
LastEditTime: 2021-09-02 21:23:56
'''
# 题目：https://leetcode-cn.com/problems/reducing-dishes/
# 思路：满意度越高的菜越靠后做，产生的价值越高。首先对满意度进行排序，满意度高的在前面，满意度低的在后面；然后依次求前缀和，如果前缀和>0，则改满意度的菜有价值，把前缀和累加到结果中
class Solution:
    def maxSatisfaction(self, satisfaction):
        # 贪心 + 排序
        # 对满意度排序，求取前缀和，判断当前新加入的菜是否有价值，
        # 结果列表每加入新数之前的数都会多加一遍
        satisfaction.sort(reverse=True)
        res, preSum = 0, 0
        for num in satisfaction:
            preSum += num
            if preSum > 0:
                res += preSum
        return res
