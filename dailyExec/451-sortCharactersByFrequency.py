'''
Description: 根据字符出现频率排序
Author: Luminary
Date: 2021-07-03 19:31:05
LastEditTime: 2021-07-03 19:31:16
'''
import collections
class Solution:
    def frequencySort(self, s: str) -> str:
        # 先将字符串转为可迭代的对象创建计数器
        str_list = list(s)
        countMap = collections.Counter(str_list)
        # 再将计数器转为可迭代的键值对元组形式根据值从大到小逆序排序
        char_tuple = countMap.items()
        res_tuple = sorted(char_tuple, key=lambda item:item[1],reverse = True)
        # 按照每一组出现的频率数将字符拼接起来
        res = ""
        for item in res_tuple:
            for _ in range(item[1]):
                res += item[0]
        return res
        