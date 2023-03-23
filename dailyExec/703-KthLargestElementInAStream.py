# 数据流中的第K大元素
import heapq
class KthLargest(object):

    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self.k = k
        self.L = nums
        # 初始化最小堆
        heapq.heapify(self.L)

    def add(self, val):
            """
        :type val: int
        :rtype: int
        """
        heapq.heappush(self.L, val)
        # 保证堆中的元素个数不超过 K个，超过就pop出堆顶元素
        while len(self.L) > self.k:
            heapq.heappop(self.L)
        # 此时堆中的最小元素（堆顶）就是整个数据流中的第 K大元素。
        return self.L[0]
