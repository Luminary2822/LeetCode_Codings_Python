'''
Description: 
Author: Luminary
Date: 2021-04-07 13:57:11
LastEditTime: 2021-04-15 17:17:36
'''
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        # 起始对整体做一次二分，判断mid与旋转点的位置对两个部分有序继续做二分
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True
            # left和right均为重复元素时，无法判断在哪个区间继续搜索，直接向右移动 left 直到他们不相等
            if nums[left] == nums[right]:
                left += 1
                continue
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
        return False


