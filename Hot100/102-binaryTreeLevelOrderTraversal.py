'''
Description: 二叉树的层序遍历
Author: Luminary
Date: 2021-05-19 20:33:22
LastEditTime: 2021-05-19 20:33:48
'''
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        # 根节点不存在特殊情况判断
        if not root: return []

        # 利用队列保存每一层的结点
        queue = [root]
        # 结果列表
        res = []

        while queue:
            # 遍历上一层结点的值以列表形式存储在res
            res.append([node.val for node in queue])
            # 存储当前层结点
            temp = []
            for node in queue:
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            # 将queue变成当前层结点
            queue = temp
        return res