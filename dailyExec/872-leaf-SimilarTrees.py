'''
Description: 叶子相似的树
Author: Luminary
Date: 2021-05-10 16:29:56
LastEditTime: 2021-05-10 16:30:43
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def leafSimilar(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: bool
        """
         # 深度优先遍历：传入根节点和结果数组
        def dfs(root, res):
            if not root:
                return 
            # 叶子节点累加其值到结果数组中
            if root.left == None and root.right == None:
                res.append(root.val)
            # 递归遍历左子树
            dfs(root.left, res)
            # 递归遍历右子树
            dfs(root.right, res)

        # 依次求取两个子树叶子节点值集，比较是否相等
        res1 = []
        dfs(root1, res1)
        res2 = []
        dfs(root2, res2)
        # 比较两个结果数组是否相等
        return res1 == res2