class Solution(object):
    def minDiffInBST(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 初始化中序遍历时前序结点
        self.pre = None
        # 保存差值，初始化结点最大值
        self.minDiff = 10e6
        # 调用中序遍历函数
        self.inorderTraversal(root)
        return self.minDiff

    def inorderTraversal(self,root):
        if not root:
            return
        # 中序遍历先访问左子树
        self.inorderTraversal(root.left)
        # 判断前序结点是否存在，如果不存在则不计算差值，说明当前遍历的是中序遍历第一个结点
        if self.pre:
            self.minDiff = min(self.minDiff, root.val - self.pre.val)
        # 将pre设置成新的前序结点
        self.pre = root
        # 遍历右子树
        self.inorderTraversal(root.right)