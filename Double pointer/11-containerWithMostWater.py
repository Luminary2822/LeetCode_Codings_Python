'''
Description: 盛最多水的容器
Author: Luminary
Date: 2021-05-18 21:44:48
LastEditTime: 2021-05-18 21:45:19
'''
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # 最优做法：双指针

        # 设置左右指针，边移动边判断
        left = 0
        right = len(height) - 1
        maxArea = 0

        while left < right:
            # 先计算长度
            w = right - left
            # 计算高度：查看左右的高度，储水量取决于最低的木板，计入h后移动指针
            if height[left] < height[right]:
                h = height[left]
                left += 1
            else:
                h = height[right]
                right -= 1
            
            # 计算当前区域面积
            area = w * h
            maxArea = max(area, maxArea)
        return maxArea