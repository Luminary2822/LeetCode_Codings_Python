'''
Description: 最高频元素的频数
Author: Luminary
Date: 2021-07-19 13:24:04
LastEditTime: 2021-07-19 13:34:53
'''
class Solution:
    def maxFrequency(self, nums, k):
        # 排序 + 滑窗
        nums.sort()
        N = len(nums)
        left,right = 0, 1
        # 最高频元素必定可以是数组中已有的某一个元素，所以初始化为 1
        res = 1
        total = 0
        # 右指针向前搜索
        while right < N:
            # 每次右移一位需要增加的数，total表示每一位都向当前right所指的值对齐的话需要走多少步长
            # 1,2,4,1向2对齐需要1步，1和2向4对齐：total已经等于1，这个1相当于1已经变成2的步长，那么2和2变成4的步长就是 2 * (4-2) = 4 再加1=5，表示1和2向4对齐所需的步长
            total += (right-left)*(nums[right] - nums[right-1])
            # 如果大于k，窗口右移，减去最左边数需要对齐到当前right需要的步长数【还回去】
            while total > k:
                total -= nums[right] - nums[left]
                left += 1
            # 实时记录窗口内的个数（窗口内都是可以在k之内对齐到当前right的）
            res = max(res, right - left + 1)
            right += 1
        return res

