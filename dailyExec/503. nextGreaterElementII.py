class Solution(object):
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 第一种方法：建立单调递减栈
        N = len(nums)
        res = [-1] * N
        # 利用栈存储数组下标，因为需要根据下标修改结果数组
        stack = []
        for i in range(N * 2):
            # 栈非空的时候判断栈顶元素下标对应数值和当前数值比较
            while stack and nums[stack[-1]] < nums[i % N]:
                # 满足条件弹出栈顶元素，根据下标记录到结果数组中
                res[stack.pop()] = nums[i % N]
            # 当前数值小于栈顶元素下标对应数值，入栈建立单调递减关系，继续寻找下一个最大的值
            # 当前栈内元素均小于栈顶元素，所以下一个最大的值均是当前栈内
            stack.append(i % N)
        return res

        
        # 第二种方法：直接题意法【超时】
        """
        N = len(nums)
        res = [-1] * N
        for i in range(N):
            # 循环数组，j走2倍的N，取模运算
            for j in range(i+1,N*2):
                # 当前位置数为下一个最大的数记录到结果数组中跳出 j 这层的循环判断下一个 i 
                if nums[i] < nums[j%N]:
                    res[i] = nums[j%N]
                    break
        return res
        """