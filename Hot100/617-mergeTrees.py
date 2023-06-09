# 合并二叉树
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


# 给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
# 你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，
# 否则不为 NULL 的节点将直接作为新二叉树的节点。

# 深度优先搜索改变二叉树的结构
class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if t1 and t2:
            t1.val += t2.val
            t1.left = self.mergeTrees(t1.left, t2.left)
            t1.right = self.mergeTrees(t1.right, t2.right)
        return t1 or t2

# 深度优先搜索不改变二叉树的结构，新建一个树
# class Solution(object):
#     def mergeTrees(self, t1, t2):
#         """
#         :type t1: TreeNode
#         :type t2: TreeNode
#         :rtype: TreeNode
#         """
#         if not t1:
#             return t2
#         if not t2:
#             return t1
        
#         merged = TreeNode(t1.val + t2.val)
#         merged.left = self.mergeTrees(t1.left, t2.left)
#         merged.right = self.mergeTrees(t1.right, t2.right)
#         return merged