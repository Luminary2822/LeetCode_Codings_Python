# 数组中的第K个最大元素
import heapq
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # 第一种：利用堆构造k个容量的优先队列
        size = len(nums)
        L = []
        for i in range(k):
            heapq.heappush(L, nums[i])
        for j in range(k,size):
            if nums[j] > L[0]:
                # 弹出最小的元素，并将x压入堆中
                heapq.heapreplace(L, nums[j])
        # 堆顶元素即为第 k 大元素
        return L[0]

        # 第二种：简单调用库函数排序获取第K大的值
        """
        size = len(nums)
        nums.sort()
        return nums[size-k]
        """
        
        # 第三种：切分操作：减而治之
        """
        # 设置左右指针和目标位置
        size = len(nums)
        left = 0
        right = size - 1
        target = size - k

        while True:
            # 进行一次切分操作，得到确定某元素的位置比较是否与目标一致
            index = self._partition(nums, left, right)
            if index == target:
                return nums[index]
            # 不一致则判断位置在目标位置左侧还是右侧以此调节指针
            elif index < target:
                left = index + 1
            else:
                right = index - 1
    # 切分函数
    def _partition(self, nums, left, right):
        pivot = nums[left]
        j = left
        for i in range(left + 1, right + 1):
            if nums[i] < pivot:
                j += 1
                nums[i], nums[j] = nums[j], nums[i]
        # 在之前遍历的过程中，满足 [left + 1, j] < pivot，并且 (j, i] >= pivot
        nums[left], nums[j] = nums[j], nums[left]
        # 交换以后 [left, j - 1] < pivot, nums[j] = pivot, [j + 1, right] >= pivot
        return j
        """





