'''
Description: 最小K个数
Author: Luminary
Date: 2021-09-03 09:37:20
LastEditTime: 2021-09-03 09:38:26
'''
import heapq
class Solution:
    def smallestK(self, arr, k) :
        # 用大根堆实时维护数组前k小值，由于python自带小根堆所以存入相反数
        if k == 0:
            return list()
        hp = [-x for x in arr[:k]]
        heapq.heapify(hp)
        for i in range(k, len(arr)):
            if -hp[0] > arr[i]:
                # 弹出最小元素并将-arr[i]压入堆中，相当于pop再push
                heapq.heapreplace(hp, -arr[i])
        return [-x for x in hp]
