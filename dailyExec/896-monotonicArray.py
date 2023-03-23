class Solution(object):
    def isMonotonic(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        # 利用两个标识来判定是否递增或者递减
        N = len(A)
        inc, dec = True,True
        for i in range(1, N):
            if A[i] < A[i-1]:
                inc = False
            if A[i] > A[i-1]:
                dec = False
            # 如果inc和dec均为False,说明数列中存在既有递增又有递减的情况，则返回False，如果数组是单调则必有一个一直为True
            if not inc and not dec:
                return False
        return True
