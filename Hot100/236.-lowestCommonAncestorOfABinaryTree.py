'''
Description: 二叉树的最近公共祖先
Author: Luminary
Date: 2021-06-20 20:32:29
LastEditTime: 2021-06-20 20:33:08
'''
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # 递归回溯
        # 终止条件：当root节点为空，或遍历到叶子节点的子节点
        if not root:
            return
        # 当root == p或root == q时，即可终止
        if root == p or root == q:
            return root
        # 从当前节点root的左子树中寻找最近公共祖先
        left = self.lowestCommonAncestor(root.left, p, q)
        # 从当前节点root的右子树中寻找最近公共祖先
        right = self.lowestCommonAncestor(root.right, p, q)
        # 如果left和right都存在，表示他们异侧
        if left and right:
            return root
        # 如果有一个为空则返回另一个
        elif left and not right:
            return left
        elif right and not left:
            return right