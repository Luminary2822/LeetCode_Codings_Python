'''
Description: 主要元素
Author: Luminary
Date: 2021-07-10 12:59:56
LastEditTime: 2021-07-10 13:00:56
'''
class Solution:
    def majorityElement(self, nums) :
        # 摩尔投票法：空间复杂度为1
        N = len(nums)
        x = -1
        count = 0
        # 遍历数组
        for num in nums:
            # count不存在给x赋予新的候选值
            if not count:
                x = num
            # 与x不同count-1相同count+1
            count += 1 if x == num else -1
        return x if count and nums.count(x) > N // 2 else -1