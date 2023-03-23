'''
Description: 反转每对括号间的子串
Author: Luminary
Date: 2021-06-01 14:19:14
LastEditTime: 2021-06-01 14:21:09
'''
class Solution(object):
    def reverseParentheses(self, s):
        """
        :type s: str
        :rtype: str
        """
        # 题意：先翻转内层，翻转结果还放回原位置和外层继续翻转
        # (ed(et(oc))el)" -> etco -> octe - > leetcode
        stack = []
        for c in s:
            # 暂时存储待反转的字符串
            temp = []
            # 没遇到右括号均压入栈中
            if c != ')':
                stack.append(c)
            else:
                # 遇到右括号，将当前括号所有值弹出存入temp
                while stack and stack[-1] != '(':
                    temp.append(stack.pop())
                # 再将左括号去除
                stack.pop()
                # 将逆序后的字符串加回栈中，等待下一次逆序（用+=加回在栈顶的末尾位置）
                stack += temp
        # 最后栈内即为结果
        return "".join(stack)
