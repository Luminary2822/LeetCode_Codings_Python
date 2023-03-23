'''
Description: 数组异或操作
Author: Luminary
Date: 2021-05-07 17:36:46
LastEditTime: 2021-05-07 19:17:46
'''
class Solution(object):
    def xorOperation(self, n, start):
        """
        :type n: int
        :type start: int
        :rtype: int
        """
        # 依据题意模拟
        res = start
        for i in range(1,n):
            res ^= start + 2 * i
        return res