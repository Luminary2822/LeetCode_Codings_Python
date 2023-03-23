class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 这个题逻辑性太强以至于不想复杂的看下去，over
        res, num, sign = 0, 0, 1
        stack = []
        for c in s:
            if c.isdigit():
                # 为什么num*10
                num = 10 * num + int(c)
            elif c == "+" or c == "-":
                res += sign * num
                num = 0
                sign = 1 if c == "+" else -1
            elif c == "(":
                stack.append(res)
                stack.append(sign)
                res = 0
                sign = 1
            elif c == ")":
                res += sign * num
                num = 0
                res *= stack.pop()
                res += stack.pop()
        # 为什么最后还要加一下呢
        res += sign * num
        return res

a = Solution()
print(a.calculate("(1+(4+5+2)-3)+(6+8)"))
