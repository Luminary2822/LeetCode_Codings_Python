'''
Description: 路径总和
Author: Luminary
Date: 2021-09-28 20:42:28
LastEditTime: 2021-09-28 20:42:28
'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def hasPathSum(self, root, targetSum) :
        # 递归判断：遍历过程中每遇到一个结点，从目标值里扣除结点值，直到叶子节点判断目标值是否被扣完。
        if root == None:
            return False 
        # 遍历到叶子节点，判断当前所需目标和-叶子节点的值是否为零满足条件
        if root.left == None and root.right == None:
            return targetSum == root.val
        return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)
        
