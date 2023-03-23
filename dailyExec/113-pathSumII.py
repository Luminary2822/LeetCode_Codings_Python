'''
Description: 路径总和II
Author: Luminary
Date: 2021-09-28 21:05:33
LastEditTime: 2021-09-28 21:05:34
'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def pathSum(self, root, targetSum):
        # 深度优先搜索:枚举根节点到叶子结点的路径，遍历到叶子结点恰好为目标和的时候，记录这条路径。
        res = []
        path = []
        def dfs(root, targetSum):
            if not root:
                return []
            path.append(root.val)
            targetSum -= root.val
            if not root.left and not root.right and targetSum == 0:
                res.append(path[:])
            dfs(root.left, targetSum)
            dfs(root.right, targetSum)
            # 清除当前选择，不影响其他路径搜索
            path.pop()
        dfs(root, targetSum)
        return res
