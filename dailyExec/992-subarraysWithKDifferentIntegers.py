import collections
class Solution(object):
    def subarraysWithKDistinct(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        # 结果 = 最多由 K 个不同整数的子数组的个数 - 最多由 K - 1 个不同整数的子数组的个数。
        return self.atMostK(A, K) - self.atMostK(A, K-1)
    # 求 A 中由最多 K 个不同整数组成的子数组的个数
    def atMostK(self, A, K):
        N = len(A)
        # 记录不同数字出现的个数
        counter = collections.Counter()
        # 左右指针
        left, right = 0, 0
        # 记录不同数字的个数
        distinct = 0
        res = 0
        while right < N:
            if counter[A[right]] == 0:
                distinct += 1
            counter[A[right]] += 1
            # 当不同数字个数大于K时，对应counter内个数减一，如果该数字个数为0则distinct减一，最后移动左指针向前
            while distinct > K:
                counter[A[left]] -= 1
                if counter[A[left]] == 0:
                    distinct -= 1
                left += 1
            # 结果集加入每个满足条件以right为右区间的子数组个数（即为长度）
            res += right - left + 1
            # right指针向前
            right += 1
        return res
    # 第二种atMostK函数：不用distinct利用k来控制个数
    """
    def atMostK(self, A, K):
        count = collections.Counter()
        res = i = 0
        for j in range(len(A)):
            if count[A[j]] == 0:
                K -= 1
            count[A[j]] += 1
            while K < 0:
                count[A[i]] -= 1
                if count[A[i]] == 0:
                    K += 1
                i += 1
            res += j - i + 1
    """