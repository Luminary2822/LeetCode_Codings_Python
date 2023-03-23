# 有效的括号
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 如果字符串长度为奇数，那一定不是
        if len(s) %2 == 1:
            return False
        # 建立一个括号映射字典
        pire = {'(':')', '{':'}', '[':']'}
        stack = []
        # 遍历字符串
        for char in s:
            # 判断是否为左括号，左括号入栈
            if char in pire:
                stack.append(char)
            # 非左括号即为右括号，弹出栈应该匹配的有括号看是否与当前字符匹配
            else:
                if not stack or pire[stack.pop()] != char:
                    return False
        # 最后若栈为空返回True，非空返回False
        return True if not stack else False

        # 一种比较简单易懂的方法，判断这些成对的括号是否出现在s中，若出现则全部替换为空，最后判断s是否为空
        """
        while '{}' in s or '()' in s or '[]' in s:
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''
        """
        