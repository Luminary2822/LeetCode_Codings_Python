'''
Description: 对称二叉树
Author: Luminary
Date: 2021-07-17 20:42:07
LastEditTime: 2021-07-17 20:45:58
'''
# 迭代（速度和空间都会更快一些）
class Solution:
    def isSymmetric(self, root):        
        # 迭代：层序遍历
        # next_queue存储下一层结点，layer存储当前层结点的值，检查每一层是不是回文数组
        queue = [root]
        while(queue):
            next_queue = list()
            layer = list()
            for node in queue:
                if not node:
                    layer.append(None)
                    continue
                next_queue.append(node.left)
                next_queue.append(node.right)       
                layer.append(node.val)
            if layer != layer[::-1]:
                return False
            queue = next_queue  
        return True
# 递归
class Solution:
    def isSymmetric(self, root) :
        # 递归判断左右子树是否对称
        def check(node1, node2):
            # 结点均不存在则返回True
            if not node1 and not node2:
                return True
            # 有一边结点不存在一边还存在返回False
            elif not node1 or not node2:
                return False
            # 判断两个结点的值是否相等，不相等返回False
            if node1.val != node2.val:
                return False
            # 递归判断node1的左子树是否与node2的右子树对称，以及node1的右子树是否与node2的左子树对称
            return check(node1.left, node2.right) and check(node1.right, node2.left)
        # 传入两个当前相同的二叉树
        return check(root, root)