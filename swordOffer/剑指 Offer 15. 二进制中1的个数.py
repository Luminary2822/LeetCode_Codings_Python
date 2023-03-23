'''
Description: 二进制中1的个数
Author: Luminary
Date: 2021-06-23 21:00:45
LastEditTime: 2021-06-23 21:01:17
'''
from typing import List
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n = n & (n - 1)
            count += 1
        return count
