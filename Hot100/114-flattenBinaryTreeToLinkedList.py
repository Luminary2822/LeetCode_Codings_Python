'''
Description: 二叉树展开为链表
Author: Luminary
Date: 2021-05-06 19:55:41
LastEditTime: 2021-05-06 19:56:12
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        if not root:
            return root
        # 递归将左右子树展开
        self.flatten(root.left)
        self.flatten(root.right)
        
        # 临时保存已经展开的右子树
        temp = root.right
        # 将已经展开的左子树接入根节点的右边
        root.right = root.left
        # 根节点左子树置为空
        root.left = None


        # 让root指向当前右子树的最后结点
        while(root.right):
            root = root.right
        # 将前面临时保存已经展开的右子树继续接入
        root.right = temp