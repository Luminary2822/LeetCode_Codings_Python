'''
Description: 将二叉搜索树按照中序遍历得到只有右孩子的树
Author: Luminary
Date: 2021-04-25 18:01:15
LastEditTime: 2021-04-25 18:08:53
'''
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def increasingBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        # 新的树的根节点
        dummpy = TreeNode(-1)
        # 保存中序遍历上一个被访问的节点，上一个被访问的节点是其左子树的最右下角的节点
        self.prev = dummpy
        # 中序遍历
        self.inOrder(root)
        return dummpy.right
    
    def inOrder(self, root):
        if not root:
            return None
        # 中序遍历左子树
        self.inOrder(root.left)

        # 以下三步保证了在中序遍历的过程中的访问顺序，形成了一个新的只有右孩子的树
        # 将当前结点左结点设置为NULL
        root.left = None

        # 将前一个访问结点的右子树设置为当前访问结点
        self.prev.right = root
        # prev移动，root变成上一个被访问的结点
        self.prev = root

        # 中序遍历右子树
        self.inOrder(root.right)