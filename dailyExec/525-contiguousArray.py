'''
Description: 连续数组
Author: Luminary
Date: 2021-06-03 22:14:50
LastEditTime: 2021-06-03 22:15:18
'''
class Solution(object):
    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 前缀和 + 哈希表

        # 将0转换为-1，相同0和相同1的个数的子数组将0换成-1之后和为0
        N = len(nums)
        for i in range(N):
            if nums[i] == 0:
                nums[i] = -1
        
        # 简历哈希表存储每个preSum[i]出现的最早下标，
        # 如果preSum[i]==preSum[j]，即说明i~j的连续子数组和为0
        map = {0:0}
        res = 0

        preSum = [0] *(N+1)
        for i in range(1, N+1):
            preSum[i] = preSum[i-1] + nums[i-1]
            # 判断是否出现相等的前缀和，出现的话计算下标之间的子数组长度更新res
            if preSum[i] in map:
                res = max(i - map[preSum[i]], res)
            else:
                map[preSum[i]] = i

        return res
