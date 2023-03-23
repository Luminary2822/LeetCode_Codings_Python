#  子数组最大平均数 I
class Solution(object):
    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        # 在python3的环境下运行结果正确
        # def findMaxAverage(self, nums: List[int], k: int) -> float:
        # 先求和，再遍历从k到最后，对滑动窗口逐个添加和删除元素再选取最大和，最后求平均
        maxSum = sum(nums[0:k])
        res = maxSum
        for i in range(k, len(nums)):
            maxSum += nums[i] - nums[i - k]
            res = max(res,maxSum)
        return res / k

