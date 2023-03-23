'''
Description: 4的幂
Author: Luminary
Date: 2021-05-31 21:52:49
LastEditTime: 2021-05-31 21:53:15
'''
class Solution(object):
    def isPowerOfFour(self, n):
        """
        :type n: int
        :rtype: bool
        """
        # 首先判断是否为2的幂
        if n < 0 or n & (n-1):
            return False
            
        # 判断n是否为4的幂，4的n即为(3+1)的n，展开多项式除结尾的1都有3相乘，所以4的幂一定可以模3余1
        # return n % 3 == 1

        # 或者判断二进制的1是否在奇数位，奇数位校验
        return n & 0x55555555 > 0