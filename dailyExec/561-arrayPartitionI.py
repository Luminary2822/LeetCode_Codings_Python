# 数组拆分
class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 数组排序，两两组合，取每对前一个累加即和最大
        res = 0
        nums.sort()
        for i in range(0,len(nums),2):
            res += nums[i]
        return res
        # 切片方法
        """
        nums.sort()
        # nums[::2]取偶数下标的数值：0,2,4,……
        return sum(nums[::2])
        """
