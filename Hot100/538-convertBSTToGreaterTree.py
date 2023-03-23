'''
Description: 把二叉树搜索树转换为累加树
Author: Luminary
Date: 2021-06-20 20:40:52
LastEditTime: 2021-06-20 20:43:39
'''
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def __init__(self):
        # num 始终保存‘比当前节点值大的所有节点值的和’
        self.num = 0
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
    # 反向中序遍历：访问的节点值是递减的，之前访问的节点值都比当前的大，每次累加给 num 即可
        if not root:
            return root
        # 递归右子树
        self.convertBST(root.right)
        # 处理当前节点
        root.val += self.num
        self.num = root.val
        # 递归左子树
        self.convertBST(root.left)
        return root