import heapq
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # 官方题解：利用堆构建优先队列
        n = len(nums)
        # 注意 Python 默认的优先队列是小根堆，构建k个数值与坐标构成元组的列表
        q = [(-nums[i], i) for i in range(k)]
        # 将列表具有堆的性质
        heapq.heapify(q)
        # 先加入-nums[0]，初始最大值
        res = [-q[0][0]]
        for i in range(k, n):
            # 队列加入一个新元素
            heapq.heappush(q, (-nums[i], i))
            # 判断当前堆顶元素（即为最大值）是否在滑动窗口内，如果不在的话它的坐标一定在滑动窗口左边界的左侧
            while q[0][1] <= i - k:
                # 后续移动滑动窗口时这个值一定不会出现了所以可以永久从优先队列删除
                heapq.heappop(q)
            # 结果集加入当前堆顶元素数值
            res.append(-q[0][0])
        return res

        # 第二种：直接理解题意法（1）(2)-会超时
        """
        result = []
        if len(nums) == 0:
            return
        # 遍历每个滑动窗口寻找最大值
        for i in range(0, len(nums)-k+1):
            result.append(max(nums[i:k+i]))
        return result
        # （2）
        if not len(nums) and not k: return []
        rst = []
        # 先构建初始窗口，选取最大值
        tmp = nums[0:k]
        rst.append(max(tmp))
        # 遍历从k开始到后面的数值，每次窗口第一个元素删除，添加一个新元素，寻找最大值加入结果列表
        for i in nums[k:]:
            del tmp[0]
            tmp.append(i)
            rst.append(max(tmp))
        return rst
        """