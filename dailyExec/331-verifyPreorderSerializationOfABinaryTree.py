class Solution(object):
    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """
        # 第一种方法：栈
        # 用栈递归判断是否为树的先序序列，主要判断叶子节点并依次消除
        stack = []
        # 以逗号将序列中分隔开加入栈中
        for node in preorder.split(','):
            # 元素先入栈
            stack.append(node)
            # 循环判断如果出现 '数字 # #'，该node一定是叶子节点，将三个元素出栈，将该节点以#代替继续判断
            while len(stack) >= 3 and stack[-1] == stack[-2] == '#' and stack[-3] != '#':
                # 三元素出栈
                stack = stack[:-3]
                # 用 '#' 代替 ' 数字 # # '
                stack.append('#')
        # 最后栈内只剩元素 # 即为一棵树的先序序列
        return len(stack) == 1 and stack[-1] == '#'

        # 第二种方法：利用树的出入度
        """
        nodes = preorder.split(',')
        diff = 1
        for node in nodes:
            diff -= 1
            if diff < 0:
                return False
            if node != '#':
                diff += 2
        return diff == 0
        """
