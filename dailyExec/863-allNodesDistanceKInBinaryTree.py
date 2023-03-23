'''
Description: 二叉树中所有距离为K的结点
Author: Luminary
Date: 2021-07-28 15:48:47
LastEditTime: 2021-07-28 15:49:17
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root, target, k) :
        # DFS深度优先遍历将树转换成图，增加parent指针
        def dfs(root, parent = None):
            if root:
                root.parent = parent
                dfs(root.left, root)
                dfs(root.right, root)
        dfs(root)

        # BFS广度优先遍历从目标结点开始寻找左右上三个方向结点计算距离，再由三个结点循环扩散开记录距离
        q = [(target, 0)]
        seen = {target}
        while q:
            # queue里面存储的元组对都是距离target有相同的距离（前一距离结点已被弹出）
            # 所以只需判断第一个是否满足，满足的话所有结点值全部输出
            if q[0][1] == k:
                return [node[0].val for node in q]
            # 弹出当前需遍历的结点和距离，以其为中心继续向左右上扩散计算
            node, distance = q.pop(0)
            for neighbor in (node.left, node.right, node.parent):
                if neighbor and neighbor not in seen:
                    seen.add(neighbor)
                    q.append((neighbor, distance + 1))
        return []
            