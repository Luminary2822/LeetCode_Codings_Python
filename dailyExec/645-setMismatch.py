'''
Description: 错误的集合
Author: Luminary
Date: 2021-07-05 11:13:45
LastEditTime: 2021-07-05 11:14:36
'''
import collections
class Solution:
    def findErrorNums(self, nums) :
        # 构建数字以及出现次数的字典
        # count_map = collections.Counter(nums)
        # for i in range(1, len(nums) + 1):
        #     if count_map[i] == 2:
        #         repeat_num = i
        #     if count_map[i] == 0:
        #         lost_num = 0
        # return [repeat_num, lost_num]

        # Hash：下标为键，出现个数为值
        hash_list = [0] * len(nums)
        for num in nums:
            hash_list[num - 1] += 1
        return [hash_list.index(2) + 1, hash_list.index(0) + 1]