'''
Description: 最长连续序列
Author: Luminary
Date: 2021-07-17 13:34:34
LastEditTime: 2021-07-17 13:52:09
'''
# 空间换时间 O(n)复杂度
class Solution:
    def longestConsecutive(self, nums) :
        # 哈希表：每次寻找能够充当左边界的元素值
        # 找到 nums 中有哪些元素能够当做连续序列的左边界
        if not nums:return 0
        nums = set(nums)
        max_length = 0
        for num in nums:
            # 如果num-1不在数组中，说明num可以充当连续序列的左边界，当前长度为1
            if num - 1 not in nums:
                current_num = num
                temp_length = 1
                # num是左边界，继续判断num+1,num+2,num+3等是否存在于集合中，并记录当前长度
                while current_num + 1 in nums:
                    current_num += 1
                    temp_length += 1
                # 每次记录最大的连续序列的长度
                max_length = max(temp_length, max_length)
        return max_length

# 排序 + 一次遍历（超过O(n)的时间复杂度）
class Solution:
    def longestConsecutive(self, nums) -> int:
        if not nums:return 0
        # 对数组排序
        nums.sort()
        res = 1
        max_res = 1
        # 遍历数组，与上一个数比较，有三种情况
        for i in range(len(nums)):
            # 相等跳过一轮循环
            if nums[i] == nums[i-1]:
                continue
            # 连续则当前长度+1，及时更新最大长度
            elif nums[i] == nums[i-1] + 1:
                res += 1
                max_res = max(res, max_res)
            # 不连续当前长度重新归1
            else:
                res = 1
        return max_res
