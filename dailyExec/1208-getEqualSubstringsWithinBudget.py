# 尽可能使字符串相等
class Solution(object):
    def equalSubstring(self, s, t, maxCost):
        """
        :type s: str
        :type t: str
        :type maxCost: int
        :rtype: int
        """
        # 问题转换为已知一个数组 costs ，求：和不超过 maxCost 时最长的子数组的长度。
        N = len(s)
        left,right = 0,0
        res = 0
        sum = 0
        costs = [0] * N
        # 计算每个字符转换对应的开销数组
        for i in range(N):
            costs[i] = abs(ord(s[i]) - ord(t[i]))
        while right < N:
            # 将当前窗口的开销累加到sum中
            sum += costs[right]
            # 判断是否超过最大值，超过则移动窗口
            while sum > maxCost:
                sum -= costs[left]
                left += 1
            res = max(res, right - left + 1)
            right += 1
        return res
