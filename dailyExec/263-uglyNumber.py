'''
Description: 
Author: Luminary
Date: 2021-04-10 16:45:57
LastEditTime: 2021-04-10 16:50:11
'''
class Solution(object):
    def isUgly(self, n):
        """
        :type n: int
        :rtype: bool
        """
        # 负数和零一定不是丑数
        if n <= 0:
            return False
        factors = [2,3,5]
        # 对质因数依次取模再反复除，直到不再包含质因数2,3,5
        for factor in factors:
            while n % factor == 0:
                n //= factor
        # 判断剩下数字是否为1，为1是丑数，否则不是
        return n == 1