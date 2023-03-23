'''
Description: 下一个最大元素I
Author: Luminary
Date: 2021-09-08 14:32:23
LastEditTime: 2021-09-08 14:32:33
'''
class Solution:
    def nextGreaterElement(self, nums1, nums2) :
        # 单调栈，构造从栈底到栈顶的单调递增栈
        stack = []
        res = [-1] * len(nums1)
        for i in range(len(nums2)):
            # 当前元素大于栈顶时，判断当前元素是否在nums1内，如果在的话则索引nums1的位置，然后记录当前元素
            while stack and nums2[i] > nums2[stack[-1]]:
                if nums2[stack[-1]] in nums1:
                    index = nums1.index(nums2[stack[-1]])
                    res[index] = nums2[i]
                stack.pop()
            # 当前元素小于等于栈顶时，直接入栈
            stack.append(i)
        return res