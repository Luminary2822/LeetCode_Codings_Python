'''
Description: 最后一个单词的长度
Author: Luminary
Date: 2021-09-21 14:32:34
LastEditTime: 2021-09-21 14:32:35
'''
class Solution:
    def lengthOfLastWord(self, s):
        # 从后向前遍历
        start = len(s) - 1
        # 确定起始点，跳过末尾的大量空格
        while s[start] == " ":
            start -= 1
        # 从起始点向前遍历最后一个单词，遇到空格就跳出循环
        res = 0
        for i in range(start, -1, -1):
            if s[i] != " ":
                res += 1
            else:
                break
        return res

