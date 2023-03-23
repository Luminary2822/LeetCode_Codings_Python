# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 二叉树的锯齿形层序遍历-BFS广度优先遍历
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root == None:
            return []
        # flag为0的层需要倒序，利用异或方法来分别
        flag = 1
        # 用来存储每一层结点的列表
        bfs = [root]
        # 结果集
        res = []
        # 当bfs非空时，获取到当前层的结点值，并且将下一层记录下来
        while bfs:
            # 用来存储下一层结点的临时列表
            temp = []
            # 存储结果集的每个小列表表示每一层的结点值
            vals = []
            for n in bfs:
                vals.append(n.val)
                if n.left:
                    temp.append(n.left)
                if n.right:
                    temp.append(n.right)
            # 根据flag判断当前层的结点值是否需要逆序
            vals = vals if flag else vals[::-1]
            # 将当前层加入结果集
            res.append(vals)
            # 更新到下一层结点继续遍历
            bfs = temp
            # 利用异或设置flag的值
            flag ^= 1
        return res
        


