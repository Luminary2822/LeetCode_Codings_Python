'''
Description: 
Author: Luminary
Date: 2021-04-08 16:49:19
LastEditTime: 2021-04-08 17:19:45
'''
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1: return nums[0]
        left, right = 0, len(nums) - 1
        # 如果left位置小于right，则说明数组有序直接返回left位置
        mid = left
        # 根据二分的中点mid的元素大小和nums[left]比较，判断mid在在旋转点之后还是之前
        # 输入的是旋转有序数组
        while nums[left] >= nums[right]:
            # 只有两个元素时，在这left一定比right大，所以最小就是right
            if left + 1 == right:
                mid = right
                break
            mid = (left + right) // 2
            # mid在旋转点左边，所以最小值在mid点右边，移动left指针
            if nums[mid] >= nums[left]:
                left = mid
            # mid在旋转点右边，所以最小值在mid点左边，移动right指针
            elif nums[mid] <= nums[right]:
                right = mid
        # mid位置元素即为最小值
        return nums[mid]
