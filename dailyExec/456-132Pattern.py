class Solution(object):
    def find132pattern(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 维护3位置，1位置找3左边的最小元素，2位置找3右边比3小的最大元素
        # 找1：左侧最小值数组 ；找3：右侧单调递减栈 + 从右向左遍历
        N = len(nums)
        # 初始化一个负无穷数组
        leftMinNum = [float("inf")] * N
        stack = []
        # 构建：任何位置左侧最小值数组，为了寻找1位置元素
        for i in range(1, N):
            leftMinNum[i] = min(leftMinNum[i-1], nums[i-1])
        for j in range(N-1,-1,-1):
            # 2号位置
            numsk = float("-inf")
            # 找到2位置元素：比当前值小的全部出栈，最后pop出的即为比3位置小的最大元素记录下来
            while stack and stack[-1] < nums[j]:
                numsk = stack.pop()
            # 判断2位置元素是否大于1位置元素
            if leftMinNum[j] < numsk:
                return True
            # 栈顶元素大于当前元素，将当前元素入栈构造一个单调递减栈
            stack.append(nums[j])
        return False
        










        # 这个题不是连续子序列，所以不能用这个方法
        # 滑动窗口方法
        N = len(nums)
        # 设置左右指针，窗口长度控制在 3 
        left, right = 0, 2
        while right < N:
            # 满足条件返回True
            if nums[left] < nums[right] < nums[left+1]:
                return True
            # 不满足窗口继续移动
            left += 1
            right += 1
        return False
