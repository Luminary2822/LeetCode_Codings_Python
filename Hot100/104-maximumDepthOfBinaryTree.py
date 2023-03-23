'''
Description: 二叉树的最大深度
Author: Luminary
Date: 2021-06-26 20:06:30
LastEditTime: 2021-06-26 20:07:34
'''
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
from typing import List
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1