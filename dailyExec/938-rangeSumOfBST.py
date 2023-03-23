'''
Description: 二叉搜索树的范围和
Author: Luminary
Date: 2021-04-28 11:59:06
LastEditTime: 2021-04-28 15:27:04
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rangeSumBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: int
        """
        # 第二种方法：
        # 输入为二叉搜索树有一定的性质，左子树一定小于root，右子树一定大于root
        res = 0
        if not root:
            return res
        # 当前结点大于low的情况下继续搜寻左子树，如果小于low则无需查找左子树，因为左子树一定小于root.val
        if root.val > low:
            res += self.rangeSumBST(root.left, low, high)
        # 在范围内满足累加res
        if low <= root.val <= high:
            res += root.val
        # 当前结点小于high的情况下继续搜寻右子树，如果大于high则无需查找右子树，因为右子树一定大于root.val
        if root.val < high:
            res += self.rangeSumBST(root.right, low, high)
        return res

        # # 第二种方法
        # # 输入为普通二叉树的递归做法
        # res = 0
        # if not root:
        #     return res
        # # 递归遍历左子树
        # res += self.rangeSumBST(root.left, low, high)
        # # 如果当前root值在low和high之间就累加
        # if low <= root.val <= high:
        #     res += root.val
        # # 再递归遍历右子树
        # res += self.rangeSumBST(root.right, low, high)
        # return res

