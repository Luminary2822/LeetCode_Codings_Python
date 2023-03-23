'''
Description: 
Author: Luminary
Date: 2021-04-16 20:47:47
LastEditTime: 2021-04-16 20:52:10
'''
class Solution(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        N = len(T)
        stack = []
        res = [0] * N
        # 建立单调递减栈存储下标
        for i in range(N):
            # 当栈存在且当前元素大于栈顶元素时，即要计算与栈内与其小的元素依次计算距离差值放入结果数组中
            while stack and T[stack[-1]] < T[i]:
                res[stack[-1]] = i - stack[-1]
                stack.pop()
            # 栈不存在或者当前元素小于栈顶元素将其入栈，寻找后面比他们大的元素
            stack.append(i)
        return res
        