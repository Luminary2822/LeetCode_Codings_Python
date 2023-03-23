'''
Description: 从前序与中序遍历序列构造二叉树
Author: Luminary
Date: 2021-05-23 21:53:02
LastEditTime: 2021-05-23 21:53:52
'''
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        # 找到根在中序中的索引loc,中序中loc，左边是root.left 右边是root.right，所以inorder[:loc]
        # 前序遍历中第一个元素是root,  root.left 有 loc个，所以 preorder[1: loc + 1]
        
        if not preorder: return None

        # 前序遍历的首元素作为根节点
        val = preorder[0]
        node = TreeNode(val)

        # 寻找根节点在中序遍历中的位置
        loc = inorder.index(val)

        
        node.left = self.buildTree(preorder[1:loc+1], inorder[:loc])
        node.right = self.buildTree(preorder[loc+1:], inorder[loc+1:])

        return node