'''
Description: 移除元素
Author: Luminary
Date: 2021-04-19 20:29:02
LastEditTime: 2021-04-19 20:31:03
'''
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        # 双指针在数组首尾，向中间遍历该序列
        left, right = 0, len(nums)-1
        while left <= right:
            # 左指针指向重复元素时，用序列右边右指针内容覆盖，然后右指针右移
            if nums[left] == val:
                nums[left] = nums[right]
                right -= 1
            # 左指针继续向前
            else:
                left += 1
        return left

        # 另一种方法
        # 如果当前元素 x 与移除元素 val 相同，那么跳过该元素。
        # idx = 0;
        # for x in nums:
        #     if (x != val):
        # 如果当前元素 x 与移除元素 val 不同，那么我们将其放到下标 idx 的位置，并让 idx 自增右移。
        #         nums[idx] = x;
        #         idx += 1
        # return idx;