class Solution(object):
    def lastStoneWeight(self, stones):
        """
        :type stones: List[int]
        :rtype: int
        """
        # 第一种：简单解法：先降序，取前两个大的相减的值再加入列表，最后剩余一个输出
        while len(stones) > 1:
            # 降序排序
            stones.sort(reverse = True)
            stones.append(stones[0] - stones[1])
            del stones[0:2]
        return stones[0]

        #  第二种：最小堆（可以自动排序）转换，python只支持小顶堆，所以在入堆的时候要添加的是数据的相反数，这样堆顶就会对应最大的元素
        """
        import heapq
        #形成一个数据相反数的列表
        heap = [-stone for stone in stones]
        #让列表具备堆特征
        heapq.heapify(heap)

        # 模拟堆的操作
        while len(heap) > 1:
            x,y = heapq.heappop(heap),heapq.heappop(heap)
            if x != y:
                heapq.heappush(heap,x-y)
        if heap: return -heap[0]
        return 0
        """
