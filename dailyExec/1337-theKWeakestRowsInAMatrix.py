'''
Description: 矩阵中战斗力最弱的K行
Author: Luminary
Date: 2021-08-01 13:34:13
LastEditTime: 2021-08-01 13:34:35
'''
class Solution:
    def kWeakestRows(self, mat, k) :
        # 先遍历后排序
        # 将每一组序号和该组1的个数组成元组形式存到res列表
        # 按照第二维升序排序，然后遍历获取前k个第一维的值
        res = []
        for i, row in enumerate(mat):
            res.append((i, row.count(1)))
        res.sort(key = lambda x:x[1])
        return [index for index, val in res[:k]]