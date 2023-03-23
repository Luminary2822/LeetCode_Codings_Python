'''
Description: 只出现一次的数字III
Author: Luminary
Date: 2021-04-30 18:22:24
LastEditTime: 2021-04-30 18:23:20
'''
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 定义哈希表
        hashmap = dict()
        res= []
        # 存储出现数字及其对应出现次数
        for i in nums:
            hashmap[i] = hashmap.get(i, 0) + 1
        # 将出现次数为1的键加入结果列表
        for i in hashmap:
            if hashmap[i] == 1:
                res.append(i)
        return res