'''
Description: 路径总和III
Author: Luminary
Date: 2021-05-23 20:21:02
LastEditTime: 2021-05-23 20:21:29
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: int
        """
        if not root:
            return 0
        self.res = 0

        # path前缀和记录
        # [10]
        # [15, 5]
        # [18, 8, 3]
        # [21, 11, 6, 3]
        # [16, 6, 1, -2]
        # [17, 7, 2]
        # [18, 8, 3, 1]
        # [7, -3]
        # [18, 8, 11]

        def dfs(root, path):
            # 记录到root结点处的所有前缀路径和，以路径上每个都为根节点计算一个
            path = [val + root.val for val in path]
            path.append(root.val)
            # 计算当前路径和中含有多少个targetSum
            self.res += path.count(targetSum)
            # 递归左子树
            if root.left:
                dfs(root.left, path)
            # 递归右子树
            if root.right:
                dfs(root.right, path)
        # 递归起点
        dfs(root,[])
        return self.res