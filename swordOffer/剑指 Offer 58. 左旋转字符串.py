'''
Description: 左旋转字符串
Author: Luminary
Date: 2021-10-03 21:09:12
LastEditTime: 2021-10-03 21:09:36
'''
class Solution:
    def reverseLeftWords(self, s, n):
        # 局部翻转+整体翻转
        # 先翻转前n，再翻转后n，最后整体翻转

        # 列表方法：切片+reverse
        s = list(s)
        s[:n] = list(reversed(s[:n]))
        s[n:] = list(reversed(s[n:]))
        s.reverse()
        return "".join(s)

        # 无调用函数方法：求余取模运算
        res = ""
        for i in range(n, n + len(s)):
            res += s[i % len(s)]
        return res