'''
Description: 矩形面积
Author: Luminary
Date: 2021-09-30 14:34:52
LastEditTime: 2021-09-30 14:34:52
'''
class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        area1 = (ax2 - ax1) * (ay2 - ay1)
        area2 = (bx2 - bx1) * (by2 - by1)
        # 计算重叠部分宽度和高度，右上角点的最小值减去左下角点的最大值
        overlapWidth = min(ax2, bx2) - max(ax1, bx1)
        overlapHeight = min(ay2, by2) - max(ay1, by1)
        overlapArea = max(overlapWidth, 0) * max(overlapHeight, 0)
        return area1 + area2 - overlapArea