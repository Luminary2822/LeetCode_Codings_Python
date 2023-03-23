# 寻找只出现一次的数（其他数均出现两次）
import collections
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 第一种方法：利用异或操作：没有使用额外空间【题目要求】
        # 0和任何数做异或运算结果仍是原来的数，自身与自身异或是0，所以连续异或下来最后剩余的数即为出现一次
        a = 0
        for num in nums:
            a = a ^ num
        return a

        # 第二种方法：使用额外空间，对nums计数   
        """                       
        countMap = collections.Counter(nums)
        N = len(nums)
        for i in range(N):
            if countMap[nums[i]] == 1:
                return nums[i]
         """ 
