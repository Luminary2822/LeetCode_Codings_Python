'''
Description: 二叉树的直径
Author: Luminary
Date: 2021-05-19 20:53:26
LastEditTime: 2021-05-19 20:54:23
'''
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 变量加上self即可在其他函数中也可以改变
        self.res = 0
        self.maxDeep(root)
        return self.res
    
    # 深度优先遍历求解：根节点对应左右结点最大深度之和的最大值
    def maxDeep(self, root):
        if not root:
            return 0
        # 遍历得到左子树的最大深度
        dl = self.maxDeep(root.left)
        # 遍历得到右子树的最大深度
        dr = self.maxDeep(root.right)
        # 求解深度之和的最大值
        self.res = max(dl+dr, self.res)
        # 返回依次为根节点所对应的最大深度
        return max(dl, dr) + 1
