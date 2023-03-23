'''
Description: 
Author: Luminary
Date: 2021-04-09 21:13:47
LastEditTime: 2021-04-11 11:04:30
'''
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 二分法
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            # mid一定在右边，移动左指针
            if nums[mid] > nums[right]: 
                left = mid + 1
            # mid一定在左边，移动右指针
            elif nums[mid] < nums[right]: right = mid
            # 当 mid和 right相等的时候，遇到重复元素移动 right指针
            else: right = right - 1
        return nums[left]

