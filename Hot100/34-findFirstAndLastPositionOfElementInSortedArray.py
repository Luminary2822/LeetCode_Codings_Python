'''
Description: 在排序数组中查找元素的第一个和最后一个位置
Author: Luminary
Date: 2021-09-01 20:41:05
LastEditTime: 2021-09-01 20:41:26
'''
class Solution:
    def searchRange(self, nums, target) :
        # 利用升序条件，使用二分查找
        # 即为寻找 [第一个等于target的位置] 和 [第一个大于target的位置-1] 
        
        # 二分查找bisect_left模块（查找元素所在位置并返回，相同元素返回应插入左边位置）
        def bisect_left(target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        # 获取target位置和target下一个数字的应插入位置-1即为开始和结束位置
        begin = bisect_left(target)
        end = bisect_left(target + 1) - 1

        # 不存在target
        if begin == len(nums) or nums[begin] != target:
            return [-1, -1]
        
        return[begin, end]
