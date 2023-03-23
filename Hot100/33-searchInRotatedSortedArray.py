'''
Description: 
Author: Luminary
Date: 2021-04-07 13:19:19
LastEditTime: 2021-04-07 14:48:10
'''
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # 起始对整体做一次二分，判断mid与旋转点的位置对两个部分有序继续做二分
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            # 判断mid在旋转点之后还是之前，分别在两个有序序列中判断target位置
            if nums[mid] <= nums[right]:
                # mid指向旋转点之后，则 mid后面是有序的，继续做一次二分
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                # mid指向旋转点之前，则 mid前面是有序的，继续做一次二分
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
        return -1


