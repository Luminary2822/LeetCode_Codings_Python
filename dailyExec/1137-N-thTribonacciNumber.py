'''
Description: 第 N 个泰波那契数
Author: Luminary
Date: 2021-09-01 20:10:33
LastEditTime: 2021-09-01 20:12:11
'''
class Solution:
    def tribonacci(self, n) :
        # 前三个数的处理
        if n == 0:  return 0
        if n <= 2:  return 1
        # 初始化
        a, b, c = 0, 1, 1
        res = 0
        # 更新abc
        for i in range(3,n+1):
            res = a + b + c
            a, b, c = b, c, res
        return res