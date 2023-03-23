'''
Description: 合并区间
Author: Luminary
Date: 2021-09-01 20:29:17
LastEditTime: 2021-09-01 20:37:29
'''
class Solution:
    def merge(self, intervals) :
        # 按照左边界排序
        intervals.sort()
        # 初始放入第一个区间
        res = [intervals[0]]
        for i in range(1, len(intervals)):
            # 判断res里区间的右边界 >= 当前区间的左边界时可以合并
            if res[-1][1] >= intervals[i][0]:
                # 每次合并都取最大的右边界，左边界不动，res内始终是合并好的区间
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                # 没有合并加入原区间到结果数组中
                res.append(intervals[i])
        return res