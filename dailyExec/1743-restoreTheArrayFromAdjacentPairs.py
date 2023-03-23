'''
Description: 从相邻元素对还原数组
Author: Luminary
Date: 2021-07-26 11:55:13
LastEditTime: 2021-07-26 11:55:58
'''
from collections import defaultdict
class Solution(object):
    def restoreArray(self, adjacentPairs):
        """
        :type adjacentPairs: List[List[int]]
        :rtype: List[int]
        """
        # 哈希表记录每个元素相邻的元素，依据哈希表的个数建立结果列表
        adjustNum = defaultdict(list)
        for i, j in adjacentPairs:
            adjustNum[i].append(j)
            adjustNum[j].append(i)
        res = [0] * len(adjustNum)

        # 遍历寻找哈希表中元素对应邻居只有一个，即为首位元素
        for key in adjustNum:
            if len(adjustNum[key]) == 1:
                res[0] = key
                break
        # 根据表中记录的邻居顺序填充，每一位让哈希表索引前一位的邻居，注意除去前面重复元素
        for i in range(1, len(res)):
            for j in adjustNum[res[i-1]]:
                if j != res[i-2]:
                    res[i] = j
        return res