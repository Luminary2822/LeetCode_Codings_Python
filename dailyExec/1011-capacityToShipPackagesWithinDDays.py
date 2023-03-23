'''
Description: 在 D 天内送达包裹的能力
Author: Luminary
Date: 2021-05-09 16:24:05
LastEditTime: 2021-05-09 16:24:47
'''
class Solution(object):
    def shipWithinDays(self, weights, D):
        """
        :type weights: List[int]
        :type D: int
        :rtype: int
        """
        # 二分查找运送所有包裹最低运载能力，最低为weights中最大值，最高为weights的和
        left = max(weights)
        right = sum(weights)

        while left < right:
            # 先判断mid运载能力
            mid = (left + right) // 2
            # 检查以该运载能力能否在D天运输完成，可以的话移动右指针继续判断更小运载能力
            if self.checksDays(weights, D, mid):
                right = mid
            # 该运载能力在D天不能完成运输的话移动左指针判断更大运载能力
            else:
                left = mid + 1
        return left

    # 检查以capability作为运载能力在D天内能否运送完所有包裹
    def checksDays(self, weights, D, capability) :
        current = 0
        day = 1
        for w in weights:
            current += w
            # 逐渐累加重量判断是否超过当前运载能力，如果超过则需要新的一天，不超过则可在1天内
            if current > capability:
                day += 1
                # 当前天的重量需要放到第二天
                current = w
        # 以当前运载能力全部运送完需要的天数与D比较，小于D说明当前能力可以，大于D说明当前不行
        return day <= D
