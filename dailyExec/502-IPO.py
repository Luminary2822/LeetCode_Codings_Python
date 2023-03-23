'''
Description: IPO
Author: Luminary
Date: 2021-09-08 14:33:03
LastEditTime: 2021-09-08 14:33:04
'''
import heapq
class Solution:
    def findMaximizedCapital(self, k, w, profits, capital) :
        # 贪心+大顶堆：每次选取所需资本最小但利益最高的项目
        N = len(profits)
        # 将profits和capital组合起来，并按本金排序，这样保证我们总能选取所有小于等于当前资本的
        project = sorted(zip(profits, capital), key = lambda x:x[1])
        idx = 0
        cur = []
        while k:
            # 将所有需要的本金小于等于当前资本的项目加入最大堆(加入为-)
            while idx < N and project[idx][1] <= w:
                heapq.heappush(cur, -project[idx][0])
                idx += 1
            # 如果当前大根堆内存在可以完成的项目，则选取堆顶最高利润项目弹出，累加到资本中。
            if cur:
                w -= heapq.heappop(cur) 
            else:
                break
            k -= 1
        return w



