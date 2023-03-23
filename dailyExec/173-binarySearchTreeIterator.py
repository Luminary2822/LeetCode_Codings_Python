class BSTIterator(object):
    
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        # 一路到底，将根节点和它的所有左节点放到栈中
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left


    def next(self):
        """
        :rtype: int
        """
        # 弹出栈顶的节点；如果它有右子树，则对右子树一路到底，把它和它的所有左节点放到栈中。
        cur = self.stack.pop()
        node = cur.right
        while node:
            self.stack.append(node)
            node = node.left
        return cur.val


    def hasNext(self):
        """
        :rtype: bool
        """
        # 判断栈内是否有元素
        return len(self.stack) > 0