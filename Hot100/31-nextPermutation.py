'''
Description: 下一个排列
Author: Luminary
Date: 2021-09-01 20:38:33
LastEditTime: 2021-09-01 20:38:55
'''
class Solution:
    def nextPermutation(self, nums) :
        """
        Do not return anything, modify nums in-place instead.
        """
        # 题意：找到一个大于当前序列的新序列，且变大的幅度尽可能小
    # 思路：寻找尽量靠右的较小数和右边尽可能小的一个较大数交换，交换完成后较大数右边需升序排列
        # 逆序寻找尽量靠右的较小数，从右向左一路上升寻找第一处断崖点i-1
        N = len(nums)
        if N <= 1:
            return 
        i = N - 1
        while i > 0 and nums[i-1] >= nums[i]:
            i -= 1
        
        # 找到断崖处i-1，将右边全部升序排列
        l = i
        r = N - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1

        # 如果下一个排列存在的话
        # 按照升序序列寻找较大数与断崖处的较小数进行交换
        if i != 0:
            for j in range(i,N):
                if nums[i-1] < nums[j]:
                    nums[i-1], nums[j] = nums[j], nums[i-1]
                    break