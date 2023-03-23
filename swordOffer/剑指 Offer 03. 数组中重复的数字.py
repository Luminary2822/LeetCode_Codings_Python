'''
Description: 数组中重复的数字
Author: Luminary
Date: 2021-10-05 14:17:49
LastEditTime: 2021-10-05 14:19:04
'''
# 时间优先：哈希表
class Solution:
    def findRepeatNumber(self, nums):
        hashTable = {}
        for num in nums:
            if num in hashTable:
                return num
            else:
                hashTable[num] = hashTable.get(num,0) + 1
        return -1

# 空间优先：原地排序
class Solution:
    def findRepeatNumber(self, nums) :
        # 通过交换操作，使元素的索引与值一一对应，通过索引映射对应的值，起到字典的作用
        # while和for不同的是，while的话交换完之后当前i不是直接加1要再走一遍循环判断是否与前面有重复
        i = 0
        while i < len(nums):
            # 数字已在对应索引位置无需交换，直接跳过
            if nums[i] == i:
                i += 1
                continue
            # 第二次遇到nums[i]，索引处已有值记录说明是重复元素，返回即可
            if nums[nums[i]] == nums[i]:return nums[i]
            # 交换至索引处
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        return -1