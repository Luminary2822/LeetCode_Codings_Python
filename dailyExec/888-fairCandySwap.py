# 公平的糖果棒交换
class Solution(object):
    def fairCandySwap(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        final_length = (sum(A) + sum(B)) / 2
        num_A = final_length - sum(A)
        for a in A:
            if num_A + a in B:
                return [a, int(a+num_A)]

a = Solution()
print(a.fairCandySwap([1,1],[3,3]))
