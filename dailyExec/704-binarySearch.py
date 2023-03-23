'''
Description: 二分查找
Author: Luminary
Date: 2021-09-06 13:05:50
LastEditTime: 2021-09-06 13:05:50
'''
class Solution:
    def search(self, nums, target):
        # 二分查找：左闭右闭
        left, right = 0, len(nums) - 1
        while left <= right:
            middle = (left + right) // 2
            if nums[middle] < target:
                left = middle + 1
            elif nums[middle] > target:
                right = middle - 1
            else:
                return middle
        return -1