class Solution(object):
    def clumsy(self, N):
        """
        :type N: int
        :rtype: int
        """
        # 遇到乘除先计算，遇到加减先入栈
        op = 0
        stack = [N]
        # 倒序遍历[N-1, 1]，运算符按顺序循环[*, / , +, -]
        for i in range(N-1, 0, -1):
            if op == 0:
                stack.append(stack.pop() * i)
            elif op == 1:
                # 注意python中的整数除法
                stack.append(int(stack.pop()/float(i)))
            elif op == 2:
                stack.append(i)
            elif op == 3:
                stack.append(-i)
            op = (op + 1) % 4
        return sum(stack)

