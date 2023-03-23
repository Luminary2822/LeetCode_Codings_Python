class Solution(object):
    def removeDuplicates(self, S):
        """
        :type S: str
        :rtype: str
        """
        stack = []
        N = len(S)
        for i in range(N):
            # 判断栈顶元素和当前字符是否相同，相同则出栈
            if stack and S[i] == stack[-1]:
                stack.pop()
            # 不同则将元素入栈
            else:
                stack.append(S[i])
        # 列表转字符串输出
        return ''.join(stack)