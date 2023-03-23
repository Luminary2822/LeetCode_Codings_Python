'''
Description: 前K个高频元素
Author: Luminary
Date: 2021-04-16 14:16:57
LastEditTime: 2021-04-16 15:11:31
'''
import heapq
import collections
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # 利用堆排序取出现次数前 k 的元素
        # 记录元素出现次数
        count = collections.Counter(nums)
        # 构造一个小顶堆：放入元素和出现次数
        heap = []
        for key,val in count.items():
            # 当堆内元素大于等于k个时判断是否需要替换
            if len(heap) >= k:
                # 当前元素值大于堆顶元素，进行替换
                if val > heap[0][0]:
                    heapq.heapreplace(heap, (val,key))
            # 堆内元素个数小于k继续push
            else:
                heapq.heappush(heap, (val,key))
        # 最后堆内元素即为次数出现前k的值键对，遍历取值
        return [item[1] for item in heap]

a = Solution()
res = a.topKFrequent([1,1,1,2,2,3],2)