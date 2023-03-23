'''
Description: 判断子序列
Author: Luminary
Date: 2021-05-15 22:08:02
LastEditTime: 2021-05-15 22:08:41
'''
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        m = len(s)
        n = len(t)
        # 定义双指针，i遍历s，j遍历t
        i,j = 0,0
        while i < m and j < n:
            # 当字符匹配上之后移动i继续判断s中下一个字符
            if s[i] == t[j]:
                i += 1
            # 如果没有匹配上或者当前匹配完成则j向后移动
            j += 1
        # 最后判断i是否移动到s串末尾
        return i == m