'''
Description: 得到子序列的最少操作次数
Author: Luminary
Date: 2021-07-26 13:50:16
LastEditTime: 2021-07-26 13:50:34
'''
import bisect
class Solution:
    def minOperations(self, target, arr) :
        # 求最长公共子序列（操作最少）等价于arr用target的坐标转换后构成最长的上升子序列
        
        # 存储target元素及其对应下标
        hashmap = {}
        for i in range(len(target)):
            hashmap[target[i]] = i 
        # 在set中查找比在list下查找要快，如果不变set的话就会超时（虽然说target中已经是不同的元素）
        target = set(target)
        
        # 将arr与target共同元素在表中遍历找到在target中的下标存在nums列表中
        nums = []
        for item in arr:
            if item in target:
                nums.append(hashmap[item])
        if not nums:
            return len(target)

        # 寻找nums列表中最长上升子序列
        d = [] 
        for num in nums:
            pos = bisect.bisect_left(d,num)
            if pos == len(d):
                d.append(num)
            else:
                d[pos] = num

        return len(target) - len(d)