'''
Description: 字符串解码
Author: Luminary
Date: 2021-09-03 16:00:09
LastEditTime: 2021-09-03 16:00:10
'''
class Solution:
    def decodeString(self, s):
        # 栈内存储信息对（左括号前的字符串，左括号前的数字）
        stack = []
        num = 0
        res = ''
        for c in s:
            # 遇到数字记录下来，
            if c.isdigit():
                # 如果是两位数比如32，那么num先记录3，再记录3*10+2
                num = num * 10 + int(c) 
            # 遇到左括号压入栈，清零开始遍历括号内的字符串
            elif c == '[':
                stack.append((res,num))
                res, num = '', 0
            # 遇到右括号，栈内弹出元素计算当前新的字符串
            elif c == ']':
                top = stack.pop()
                res = top[0] + res * top[1]
            # 遇到字符直接记录
            else:
                res += c
        return res
