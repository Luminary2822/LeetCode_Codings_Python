'''
Description: 二叉树的垂序遍历
Author: Luminary
Date: 2021-07-31 17:51:39
LastEditTime: 2021-07-31 17:52:15
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import defaultdict
class Solution:
    def verticalTraversal(self, root) :
        # 纵坐标为键，横坐标和结点值组成元组作为值，用哈希表记录坐标位置，用深度优先搜索进行遍历
        hashmap = defaultdict(list)
        def dfs(node, x, y):
            if not node:
                return 
            hashmap[y].append((x, node.val))
            dfs(node.left, x + 1, y - 1)
            dfs(node.right, x + 1, y + 1)
        dfs(root, 0, 0)

        # 先按照key排序，每个key下：按照value的第一维横坐标进行升序直接sorted，输出value的第二维结点值
        res = []
        for i in sorted(hashmap.keys()):
            temp = []
            for _,val in sorted(hashmap[i]):
                temp.append(val)
            res.append(temp)
        return res
        # 将上诉浓缩成return [[val for _, val in sorted(hashmap[x])] for x in sorted(hashmap)]