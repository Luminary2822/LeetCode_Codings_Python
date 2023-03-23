import operator
class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        # def evalRPN(self, tokens: List[str]) -> int:这题用python3交的，python2有用例错误
        # 设置操作符字典，每个操作符对应不同的运算操作
        operators = {'+':operator.add, '-':operator.sub,
                     '*':operator.mul, '/':lambda a,b:int(a/b)}
        stack = []
        for c in tokens:
            # 遇到操作符则弹出栈顶两个数字进行运算，将运算结果入栈
            if c in operators:
                a = stack.pop()
                b = stack.pop()
                # 注意先弹出的是除数，后弹出的是被除数
                stack.append(operators[c](b,a))
            # 非操作符则直接入栈
            else:
                stack.append(int(c))
        # 最后栈内元素即为表达式最后的结果
        return stack[-1]

a = Solution()
print(a.evalRPN(["4","13","5","/","+"]))
                
