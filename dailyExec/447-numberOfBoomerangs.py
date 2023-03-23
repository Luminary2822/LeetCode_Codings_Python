'''
Description: 回旋镖的数量
Author: Luminary
Date: 2021-09-13 20:10:16
LastEditTime: 2021-09-13 20:10:17
'''
class Solution:
    def numberOfBoomerangs(self, points):
        # 哈希表存储每个点作为i点时，其他点距离i点的(距离：个数)
        # 对每个点都要单独建立一个哈希表，计算其余各点到该点的距离，存储到表中
        res = 0
        for pi in points:
            hashTable = dict()
            for pj in points:
                if pi == pj:continue
                # 计算以pi为起点的距离，将根号改为平方
                dist = (pi[0] - pj[0])**2 + (pi[1] - pj[1])**2
                hashTable[dist] = hashTable.get(dist,0) + 1
            # 相同距离的个数可以组成的三元组有n * (n-1)个【排列组合】
            for value in hashTable.values():
                res += value * (value - 1)
        return res
        