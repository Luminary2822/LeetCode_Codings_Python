'''
Description: 字符串的排列
Author: Luminary
Date: 2021-06-22 13:11:47
LastEditTime: 2021-06-22 13:12:32
'''
class Solution:
    def permutation(self, s: str):
        # 回溯法
        res = []
        def backtrace(s, path):
            # 这种去重会超时，用set去重就不会超时
            # if not s and path not in res:
            if not s:
                res.append(path)
            for i in range(len(s)):
                # 当前选择的s[i]元素和其他元素的全排列组合，继续回溯
                backtrace(s[:i] + s[i+1:], path + s[i])
        backtrace(s, "")
        return list(set(res))
