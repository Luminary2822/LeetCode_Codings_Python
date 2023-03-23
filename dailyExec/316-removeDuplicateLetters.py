import collections
class Solution(object):
    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        # 一个用列表实现的栈,pop和append都是列表的
        stack = []
        c = collections.Counter(s)
        for i in s:
            if i not in stack:
                while stack and stack[-1] > i and c[stack[-1]] > 0:
                    stack.pop()
                stack.append(i)
            c[i] -= 1
        return ''.join(stack)
        