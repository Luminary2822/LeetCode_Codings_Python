'''
Description: 罗马数字转整数
Author: Luminary
Date: 2021-05-15 21:13:37
LastEditTime: 2021-05-15 21:14:00
'''
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        map = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
        res = 0
        for i in range(len(s)):
            # 小的数字在大的数字左边特殊情况
            if i < len(s) - 1 and map[s[i]] < map[s[i+1]]:
                # 如果是IV这种，在遍历到I的时候res是-I，遍历到V的时候是加V，所以最后res是V-I
                res -= map[s[i]]
            # 遍历最后一个数字或者当前数字大于后面数字的时候累加
            else:
                res += map[s[i]]
        return res