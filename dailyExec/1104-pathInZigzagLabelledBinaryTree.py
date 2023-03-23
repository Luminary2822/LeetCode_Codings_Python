'''
Description: 二叉树寻路
Author: Luminary
Date: 2021-07-29 21:01:34
LastEditTime: 2021-07-29 21:02:00
'''
class Solution:
    def pathInZigZagTree(self, label: int):
        # 顺序完全二叉树已知目标结点label求父节点（node//2）路径
        res = []
        while label != 1:
            res.append(label)
            label //= 2
        # 最后添加根节点且调整顺序为正序
        res.append(1)
        res.reverse()

        # 对于当前路径：从倒数第二个开始，每隔一个，找出取反相对应的值
        # 取反公式：本行最大值 - （当前值 - 本行开始值）
        # 完全二叉树每一层由 2**n (n从0开始)开始，最大值为 2**(n + 1)[下一层的起始] - 1
        for i in range(len(res) - 2, -1 ,-2):
            origin = res[i]
            start = 2 ** i
            end = 2 ** (i + 1) - 1
            new = end - (origin - start)
            res[i] = new
        return res
