'''
Description: 前K个高频单词
Author: Luminary
Date: 2021-05-22 21:10:36
LastEditTime: 2021-05-23 14:58:44
'''
import collections
import heapq
class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        # 先统计每个单词出现的次数
        count = collections.Counter(words)
        # 按照value降序key升序排序形成新列表，（出现次数相同按照单词首字母排序小的要在前面）
        count1 = sorted(count.items(), key = lambda x :(-x[1], x[0]))

        # 构造小顶堆
        heap = []
        # 遍历键值对，存储到堆中，堆中保持k个元素
        for key,val in count1:
            # 当此时堆中已有k个元素或者大于k个元素时，
            if  len(heap) >= k:
                # 比较当前值与堆顶，大于堆顶则弹出堆顶将新元素压入堆中，保证堆内是前k个最大值
                if val > heap[0][0]:
                    heapq.heapreplace(heap, (-val,key))
            # 当前堆小于k个元素继续入堆
            else:
                heapq.heappush(heap, (val, key))
        
        # 将小顶堆按照val降序排序，按key升序
        heap.sort(key=lambda x:(-x[0], x[1]))
        # 输出堆内元素的第二维key
        return [item[1] for item in heap]