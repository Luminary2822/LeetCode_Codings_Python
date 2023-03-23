class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 最佳方法：利用哈希表建立值和位置的对应关系，再查找nums2的时候可以降低时间复杂度
        dict = {}
        # 遍历列表
        for index,nums in enumerate(nums):
            # 寻找nums2是否在字典中，在的话将其作为key找到位置
            nums2 = target - nums
            if nums2 in dict:
                return [dict[nums2],index]
            else:
                # 建立数值和位置构成的字典键值对
                dict[nums] = index
        # 暴力遍历：时间复杂度O(N²)
        """
        N = len(nums)
        for i in range(N):
            for j in range(i,N):
                if nums[i] + nums[j] == target:
                    return[i,j]
        """