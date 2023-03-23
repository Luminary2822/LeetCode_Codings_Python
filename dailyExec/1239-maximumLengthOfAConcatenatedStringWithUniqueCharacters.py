'''
Description: 
Author: Luminary
Date: 2021-06-19 15:26:00
LastEditTime: 2021-06-19 15:29:18
'''
class Solution(object):
    def maxLength(self, arr):
        """
        :type arr: List[str]
        :rtype: int
        """
        # 回溯法
        self.res = 0
        self.backtrace(0, arr, '')
        return self.res
    
    def backtrace(self, index, arr, temp):
        # 判断temp中是否含有重复字符，没有的话记录最长长度
        if len(temp) == len(set(temp)):
            self.res = max(self.res, len(temp))
        # 回溯
        for i in range(index, len(arr)):
            self.backtrace(i+1, arr, temp + arr[i])