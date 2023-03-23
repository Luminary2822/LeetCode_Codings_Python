'''
Description: 三数之和
Author: Luminary
Date: 2021-06-18 20:03:47
LastEditTime: 2021-06-18 20:04:08
'''
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # 三指针
        n = len(nums)
        nums.sort()
        res = []
        # 枚举第一个元素
        for first in range(n-2):
            # 数组里最小的都大于0 不可能有答案
            if nums[first] > 0: break
            # 保证first不会有重复
            if first > 0 and nums[first] == nums[first-1]:continue
            # 标准双指针写法
            second, third = first + 1, n - 1
            while second < third:
                target = -nums[first]
                sum = nums[second] + nums[third]
                # 当前数值太大 做出决策：右指针左移
                if sum > target:
                    third -= 1
                # 当前数值太大 做出决策：右指针左移
                elif sum < target:
                    second += 1
                # 数值正合适 做出决策：左指针右移且右指针左移 注意不能重复
                else:
                    res.append([nums[first], nums[second], nums[third]])
                    second += 1
                    third -= 1
                    while third > second and nums[third] == nums[third+1]: third -= 1
                    while third > second and nums[second] == nums[second-1]: second += 1
        return res