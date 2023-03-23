'''
Description: 和为K的子数组
Author: Luminary
Date: 2021-09-01 20:49:02
LastEditTime: 2021-09-01 20:49:24
'''
class Solution:
    def subarraySum(self, nums, k) :
        # 将连续子数组之和表示为前缀和之差pre[i]-pre[j] (j<i)
        # 前缀和 + 哈希表存储前缀和出现的次数
        pre, res, hashTable = 0, 0, {}
        # 当数组中有0的时候，不设置这个就会少计数一次：例如{1,1,0} k = 2的时候应该是2,1+1和1+1+0
        hashTable[0] = 1
        for i in range(len(nums)):
            pre += nums[i]
            if pre - k in hashTable:
                res += hashTable[pre - k]
            if pre not in hashTable:
                hashTable[pre] = 0
            hashTable[pre] += 1
        return res