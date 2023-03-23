# 最大连续1的个数III：最多可以把 K 个 0 变成 1，求仅包含 1 的最长子数组的长度
# 题意转换：找出一个最长的子数组，该子数组内最多允许有 K个0 
class Solution(object):
    def longestOnes(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        left,right = 0,0
        res = 0
        zeros = 0
        N = len(A)
        while right < N:
            if A[right] == 0:
                zeros += 1
            # 记录窗口内0的个数与K比较，大于K则移动左指针，同时判断最左元素是否为0
            while zeros > K:
                if A[left] == 0:
                    zeros -= 1
                left += 1
            res = max(res, right - left + 1)
            right += 1
        return res
