class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # set() 函数创建一个无序不重复元素集
        new_nums = list(set(nums))
        # 判断set和原数组长度是否相同，不同说明有重复元素返回true，相同说明没有重复元素返回false
        return len(new_nums) != len(nums)
        # 这里直接一行解决问题
        # return len(set(nums)) != len(nums)