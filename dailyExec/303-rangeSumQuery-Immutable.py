class NumArray(object):
    
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        N = len(nums)
        self.preSum = [0] * (N+1)
        # 计算前缀和数组
        for i in range(N):
            self.preSum[i+1] = self.preSum[i] + nums[i]
    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.preSum[j+1] - self.preSum[i]