'''
Description: 3的幂
Author: Luminary
Date: 2021-09-23 13:07:25
LastEditTime: 2021-09-23 13:07:26
'''
class Solution:
    def isPowerOfThree(self, n) :
        if n == 1:  return True
        if n == 0:  return False
        while n != 1:
            if n % 3 == 0:
                n //= 3
            else:
                return False
        return True