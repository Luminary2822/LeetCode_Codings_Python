# 最大连续1的个数
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 第一种方法：数组一次遍历官方题解
        count = 0
        maxNum = 0
        for num in nums:
            if num == 1:
                # 记录当前连续1的个数
                count += 1
            else:
                # 使用之前连续1的个数更新最大连续1的更熟，将当前连续1的个数清零。
                maxNum = max(maxNum, count)
                count = 0
        # 数组的最后一个元素可能是1，最长连续1的子数组可能出现在数组末尾，所以遍历完需要再更新一遍
        maxNum = max(maxNum, count)
        return maxNum

        # 第二种方法滑动窗口，记录遇到0的元素位置
        """
        N = len(nums)
        res = 0
        index = -1
        # 利用index记录0出现的位置
        for i, num in enumerate(nums):
            if num == 0:
                index = i
            else:
                res = max(res, i - index)
        return res
        """