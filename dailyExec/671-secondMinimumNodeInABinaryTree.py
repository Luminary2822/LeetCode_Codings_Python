'''
Description: 二叉树中第二小的节点
Author: Luminary
Date: 2021-07-27 17:17:33
LastEditTime: 2021-07-27 17:17:56
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findSecondMinimumValue(self, root):
        # 二叉树根节点x的值小于以x为根的子树中所有节点的值
        # 深度优先遍历，寻找严格比根节点小的值
        
        res = self.dfs(root, root.val)
        # if res == float('inf'):
        #     return -1
        # else:
        #     return res
        return res if res != float('inf') else -1

    def dfs(self, root, min_):
        if not root:
            return float('inf')
        if root.val > min_:
            return root.val
        return min(self.dfs(root.left, min_), self.dfs(root.right, min_))