# 翻转二叉树
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root == None:
            return root
        # 先交换左右子树
        root.left, root.right = root.right, root.left
        # 递归交换左右子树
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root