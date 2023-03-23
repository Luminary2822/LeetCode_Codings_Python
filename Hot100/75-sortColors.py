'''
Description: 颜色分类
Author: Luminary
Date: 2021-04-23 15:50:03
LastEditTime: 2021-04-23 15:58:17
'''
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # 三路排序：左右双指针，i遍历当前数组，
        N = len(nums)
        i = 0
        left = 0
        right = N - 1
        
        # i在left到right范围内时：
        # 为0放到左指针位置，为2放到右指针位置，这两种情况i指针不移动因为交换过来的数还未确定
        while i < N:
            if nums[i] == 0 and i > left:
                nums[i], nums[left] = nums[left], nums[i]
                left += 1
            elif nums[i] == 2 and i < right:
                nums[i], nums[right] = nums[right], nums[i]
                right -= 1
            else:
                i += 1