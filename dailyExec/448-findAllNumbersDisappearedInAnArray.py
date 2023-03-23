class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for num in nums:
            # 作为下标需要-1，将对应位置数变为负数
            nums[abs(num)-1] = -abs(nums[abs(num)-1])
            # 还有正数说明对应位置的数字缺失，位置到数字需要+1
        return [i+1 for i,num in enumerate(nums) if num>0]
        # 简易版：利用set去重
        """
        # 构造全部数字的set，减去原set差即为缺失的数字
        return set([i for i in range(1,len(nums)+1)])-set(nums)
        """