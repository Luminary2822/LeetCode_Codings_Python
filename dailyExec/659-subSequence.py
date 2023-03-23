# 升序数组分割一个或者多个子序列，每个子序列将由征连续整数且长度至少为3
# 哈希表 + 最小堆
import collections
import heapq
class Solution(object):
    def isPossible(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 创建一个列表字典键为子序列结尾值，值为子序列的长度（用最小堆存储，堆顶为最小子序列长度）
        # defaultdict 创建默认值
        mp = collections.defaultdict(list)
        for x in nums:
            # 如果存在以x-1为结尾的子序列，queue为以x-1为结尾的子序列对应的最小堆，存储着不同的长度
            # python3.8可以用海象运算符：queue := mp.get(x-1)
            queue = mp.get(x - 1)
            if queue :
                # 获取到以x-1结尾的最小子序列长度
                prevLength = heapq.heappop(queue)
                # 以x结尾的子序列的最小堆，push进一个长度值+1
                heapq.heappush(mp[x], prevLength + 1)
            else:
                # 不存在以x-1为结尾的子序列就创造一个以x为结尾的子序列，对应的值最小堆为 1
                heapq.heappush(mp[x], 1)
        # 遍历每个最小堆（判断子序列的存在和最小长度与3的比较）：
        # 每个最小堆满足：存在，且堆顶元素[0]大于3返回False
        # 所有最小堆都满足为False，经过any为False，再经过not为True表示分割成功
        # 若有一个最小堆不满足为True，经过any为True，再经过not为False表示分割失败
        return not any(queue and queue[0] < 3 for queue in mp.values())

a = Solution()
print(a.isPossible([1,2,3,3,4,4,5,5]))


                
                
        