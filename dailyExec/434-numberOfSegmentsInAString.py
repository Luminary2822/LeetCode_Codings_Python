'''
Description: 字符串中的单词数
Author: Luminary
Date: 2021-10-07 13:36:07
LastEditTime: 2021-10-07 13:36:07
'''
class Solution:
    def countSegments(self, s):
        # i = 0表示记录第一个字符不是空格的时候，累加第一个字符
        # 只要s[i]是空格，s[i-1]不是空格，res就加1
        res = 0
        for i in range(len(s)):
            if s[i] != ' ' and (i == 0 or s[i-1] == ' '):
                res += 1
        return res
