'''
Description: 搜索插入位置
Author: Luminary
Date: 2021-09-27 20:37:22
LastEditTime: 2021-09-27 20:37:22
'''
class Solution:
    def searchInsert(self, nums, target):
        # 二分查找左闭右闭
        left, right = 0, len(nums) - 1
        # (1)数组中找到目标值的情况直接返回下标
        while left <= right:
            middle = (left + right) // 2
            if target > nums[middle]:
                left = middle + 1
            elif target < nums[middle]:
                right = middle - 1
            else:
                return middle
        # 分别处理如下三种情况:
        # (2)目标值在数组所有元素之前 [0,-1] 
        # (3)目标值插入数组中的位置 [left, right] ，return right + 1 或者 return left 即可
        # (4)目标值在数组所有元素之后的情况 [left, right]，return right + 1 或者 return left 即可
        # left = right + 1
        return right + 1
