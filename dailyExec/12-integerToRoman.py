'''
Description: 整数转罗马数字
Author: Luminary
Date: 2021-05-15 21:04:41
LastEditTime: 2021-05-15 21:05:02
'''
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        # 使用哈希表，按照从大到小顺序排列
        hashmap = {1000:'M', 900:'CM', 500:'D', 400:'CD', 100:'C', 90:'XC', 50:'L', 40:'XL', 10:'X', 9:'IX', 5:'V', 4:'IV', 1:'I'}
        res = ''
        for key in hashmap:
            # 比当前数字小的最大值可用来表示num
            if num // key != 0:
                # 计算有几个key
                count = num // key
                # 加入结果中
                res += hashmap[key] * count
                # key用完后减少num
                num %= key
        return res