'''
Description: 平方数之和
Author: Luminary
Date: 2021-04-28 16:24:52
LastEditTime: 2021-04-28 16:25:24
'''
import math
class Solution(object):
    def judgeSquareSum(self, c):
        """
        :type c: int
        :rtype: bool
        """
        # 双指针，注意j的最大值可以设置为c的平方根
        i = 0
        j = int(math.sqrt(c))
        # 利用双指针不断判断是否平方和等于c
        while i <= j:
            res = i * i + j * j
            if res == c:
                return True
            # 当前结果小于c移动左指针，增加res
            elif res < c:
                i += 1
            # 当前结果大于c移动右指针，减小res
            elif res > c:
                j -= 1
        return False