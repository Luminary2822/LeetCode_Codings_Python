'''
Description: 表现良好的最长时间段
Author: Luminary
Date: 2021-07-22 14:50:50
LastEditTime: 2021-09-02 21:23:49
'''
# 题目：https://leetcode-cn.com/problems/longest-well-performing-interval/
# 暴力解法：将大于8转换成1，小于8转换成-1，然后计算新数组的加和 > 0满足条件记录长度
class Solution:
    def longestWPI(self, hours):
        temp = []
        res = 0
        for hour in hours:
            if hour > 8:
                temp.append(1)
            else:
                temp.append(-1)
        for i in range(len(temp)):
            temp_sum = 0
            for j in range(i, len(temp)):
                temp_sum += temp[j]
                if temp_sum > 0:
                    res = max(res, j - i + 1)
        return res

# 前缀和 + 单调栈
class Solution:
    def longestWPI(self, hours):
        length = len(hours)
        """ 
        将最初的时间转化为是否是劳累的一天存储于数组score中（劳累记为1，不劳累记为-1）
        原问题转化为寻找连续和 >0 的最长子串长度
        """
        score = [0] * length
        for i in range(length):
            if hours[i] > 8:
                score[i] = 1
            else:
                score[i] = -1

        """
        计算前缀和，通过前缀和的差值来判断是否是良好的表现时间段
        问题转化为：寻找一对i,j;使presum[j] > presum[i],并且j-i最大，即转化为求最长上坡路问题
        """
        presum = [0] * (length + 1)
        for i in range(1, length + 1):
            presum[i] = presum[i - 1] + score[i - 1]

        """
        用单调递减栈存储presum中的元素的位置，如果理解为上坡问题的话，单调栈中维护的元素从底到顶高度依次降低，
        即越来越深入谷底，遍历完成后的栈顶位置的元素即所有元素中谷底的高度
        """
        stack = []
        for i in range(length + 1):
            if not stack or presum[i] < presum[stack[-1]]:
                stack.append(i)

        """
        从尾部遍历presum，如果该位置元素比stack中存储的位置的元素高，则表明为上坡路，弹出栈顶元素，并记录坐标差，
        该坐标差即为上坡路的长度
        """
        ans = 0
        for i in range(length, -1, -1):
            while stack and presum[i] > presum[stack[-1]]:
                ans = max(ans, i - stack[-1])
                stack.pop()
        return ans

# 哈系表方法
class Solution:
    # https://www.bilibili.com/video/BV1Wt411G7vN?from=search&seid=17906628306756620390
    def longestWPI(self, hours):
        # 哈希表：记录负数和第一次出现的位置，相同sum的最小索引
        # 判断往前少一步的前缀和是否存在，存在则计算一下距离，不用考虑多两步或者多三步，因为一定包含在一步内
        hashMap = dict()
        res = 0
        preSum = 0
        for i in range(len(hours)):
            preSum += 1 if hours[i] > 8 else -1
            # 如果当前preSum > 0，则[0-i]一定是满足条件的区间，记录长度为i + 1
            if preSum > 0:
                res = i + 1
            else:
            # 如果当前前缀和不存在字典中则记录当前的下标，即为当前preSum的最小索引；存在过就不放
                if preSum not in hashMap:
                    hashMap[preSum] = i
                # 判断-1是否存在，如果存在计算一下距离
                if preSum - 1 in hashMap:
                    res = max(res, i - hashMap.get(preSum - 1))
        return res
a = Solution()
print(a.longestWPI([9,9,6,0,6,6,9]))