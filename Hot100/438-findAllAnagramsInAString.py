'''
Description: 找到字符串中所有字母异位词
Author: Luminary
Date: 2021-09-01 20:51:51
LastEditTime: 2021-09-01 20:52:43
'''
class Solution:
    def findAnagrams(self, s, p) :
        # 滑动窗口 + 数组
        
        m, n, res = len(s), len(p), []
        if m < n:return res
        # 用数组记录字母出现的次数
        books_ord = [0] * 26
        bookp_ord = [0] * 26
        # 记录n个字母频次
        for i in range(n):
            books_ord[ord(s[i]) - ord('a')] += 1
            bookp_ord[ord(p[i]) - ord('a')] += 1
        
        # 首个窗口判断
        if books_ord == bookp_ord:
            res.append(0)
        # 剩余窗口判断
        for i in range(n,m):
            # 增加当前遍历字符，去除左边窗口旧字符
            books_ord[ord(s[i])-ord('a')] += 1
            books_ord[ord(s[i-n])-ord('a')] -= 1
            if books_ord == bookp_ord:
                res.append(i - n + 1)
        return res