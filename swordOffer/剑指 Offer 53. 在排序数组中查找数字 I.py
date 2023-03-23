'''
Description: 在排序数组中查找数字I
Author: Luminary
Date: 2021-07-16 11:51:12
LastEditTime: 2021-07-16 13:28:50
'''
class Solution:
    def search(self, nums, target):
        # 二分法，从左右两方向向target逼近
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right-left) // 2
            if nums[mid] > target:
                right = mid - 1
            if nums[mid] < target:
                left = mid + 1
        
            # 先用二分法找到mid为target时，left和right向target靠近，寻找左右边界
            if nums[mid] == target:
                if nums[left] == target and nums[right] == target:
                    return right - left + 1
                if nums[left] < target:
                    left += 1
                if nums[right] > target:
                    right -= 1
                
        return 0

# 两次二分，分别找到重复元素的左边界和右边界
# class Solution:
#     def search(self, nums, target):
#         n = len(nums)
#         if not n:
#             return 0
#         a = b = -1

#         # 二分法找到左边界
#         l, r = 0, n - 1
#         while l < r:
#             mid = l + r >> 1
#             if nums[mid] >= target:
#                 r = mid
#             else:
#                 l = mid + 1
        
#         if nums[r] != target:
#             return 0
#         a = r

#         # 二分法找到右边界
#         l, r = 0, n - 1
#         while l < r:
#             mid = l + r + 1 >> 1
#             if nums[mid] <= target:
#                 l = mid
#             else:
#                 r = mid - 1
        
#         if nums[r] != target:
#             return 0
#         b = r
#         return b - a + 1

a = Solution()
print(a.search([5,7,7,8,8,10],8))