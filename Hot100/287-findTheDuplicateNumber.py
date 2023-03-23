'''
Description: 寻找重复数
Author: Luminary
Date: 2021-04-18 20:24:03
LastEditTime: 2021-04-18 20:24:24
'''
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 定义一个哈希表
        hashtable = dict()
        for num in nums:
            # 查找数组中的键
            # 如果没有的话返回默认值0，将其值更新为1插入到字典当中，如果有的话则在其值1的基础上+1
            hashtable[num] = hashtable.get(num,0) + 1
            # 当某个键的值为2的时候，说明为重复值
            if hashtable[num] == 2:
                return num