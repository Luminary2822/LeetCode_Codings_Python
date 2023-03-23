'''
Description: 只出现一次的数字II
Author: Luminary
Date: 2021-04-30 14:07:47
LastEditTime: 2021-04-30 14:08:15
'''
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 定义一个哈希表存储出现数字和对应次数
        hashTable = dict()
        
        for num in nums:
            # hashTable.get(num,0)如果没有该键的话将值设为0并返回默认值0
            # 出现num将其值更新加1插入到字典当中
            hashTable[num] = hashTable.get(num,0) + 1
        
        # 遍历字典寻找value值为1的key
        for key,value in hashTable.items():
            if value == 1:
                return key