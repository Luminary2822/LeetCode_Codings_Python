'''
Description: 最长有效括号
Author: Luminary
Date: 2021-04-15 18:56:54
LastEditTime: 2021-04-15 20:44:42
'''
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack = []
        res = 0
        N = len(s)
        for i in range(N):
            # 将元素下标入栈：栈为空或当前元素为左括号或栈顶元素为右括号三种情况都不能进行配对消除
            if not stack or s[i] == '(' or s[stack[-1]] == ')':
                stack.append(i)
            else:
                # 表明当前有效可以匹配
                stack.pop()
                # 这里为什么先弹出后减是要匹配最远的也就是最先出栈的，它当初的位置在当前栈顶位置之后
                # 样例：")()())"，如果先计算长度的话结果就是2，因为计算的是第二个完整括号的长度，如果先弹出后计算的话，第二个完整右括号减去第一个当前栈顶右括号即为当前右括号到第一个完整括号的左括号匹配长度
                # 我说的有点啰嗦我懂就好不我以后一定不懂
                res = max(res, i - (stack[-1] if stack else -1))
        return res
a = Solution()
print(a.longestValidParentheses(")()())"))