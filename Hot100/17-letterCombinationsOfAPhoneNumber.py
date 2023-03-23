'''
Description: 电话号码的字母组合
Author: Luminary
Date: 2021-05-24 21:12:55
LastEditTime: 2021-05-24 21:13:25
'''
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits: return []
        # 构建字典
        phone = {'2':['a','b','c'],
                 '3':['d','e','f'],
                 '4':['g','h','i'],
                 '5':['j','k','l'],
                 '6':['m','n','o'],
                 '7':['p','q','r','s'],
                 '8':['t','u','v'],
                 '9':['w','x','y','z']}

        # 定义回溯函数，当nextdigit非空时，对于 nextdigit[0] 中的每一个字母 letter，执行回溯
        def backtrace(path, nextdigits):
            # 直至 nextdigit 为空。最后将 path 加入到结果中。
            if len(nextdigits) == 0:
                res.append(path)
            else:
                for word in phone[nextdigits[0]]:
                    backtrace(path + word, nextdigits[1:])
        res = []
        backtrace('', digits)
        return res