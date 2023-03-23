# 最长湍流子数组
# 湍流子数组：元素的值的变化「减少」和「增加」交替出现，且相邻元素的值不能相等。
class Solution(object):
    def maxTurbulenceSize(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        # 第一种双指针法：用数字表明方向，利用前后两方向乘积判断当前是否为湍流数组
        # 进而判断平坡还是方向相同来移动左指针，右指针稳步向前，实时更新前置方向和最大长度
        n = len(arr)
        left = 0
        # 数组长度为 1是无方向
        direction = 0
        maxLen = 1
        for right in range(1, n):
            # 如果不是湍流数组，那么前后两个方向乘积定>=0
            # 当前后方向相同或者平坡时（存在相邻的相同元素）
            if (arr[right] - arr[right - 1]) * direction >= 0:
                # 平坡，移动left指针越过重复元素到right
                if arr[right] == arr[right - 1]:
                    left = right
                # 方向相同，移动left到新方向的起点处
                else:
                    left = right - 1
            # 更新前置方向
            direction = arr[right] - arr[right-1]
            # 获取当前最大长度
            maxLen = max(maxLen, right - left + 1)
        return maxLen
        # 第二种：动态规划
        # 状态 dp[i] 为：以 i 位置结尾的最长连续子数组的长度
        # 定义两个状态数组，分别表示以 i 结尾的在增长和降低的最长湍流子数组长度，初始化为 1
        # 状态定义：
        # 定义 up[i] 表示以位置 i 结尾的，并且 arr[i - 1] < arr[i] 的最长湍流子数组长度。
        # 定义 down[i] 表示以位置 i 结尾的，并且 arr[i - 1] > arr[i] 的最长湍流子数组长度
        # 状态转移方程定义：
        # up[i] = down[i - 1] + 1，当 arr[i - 1] < arr[i]；
        # down[i] = up[i - 1] + 1，当 arr[i - 1] > arr[i]
        """
        N = len
        # 定义两个状态数组，初始化为 1
        up = [1] * N
        down = [1] * N
        res = 1
        for i in range(1, N):
            # 状态转移方程
            if arr[i-1] < arr[i]:
                up[i] = down[i-1] + 1
            elif arr[i - 1] > arr[i]:
                down[i] = up[i - 1] + 1
            # 获取最大长度
            res = max(res, max(up[i], down[i]))
        return res
        """
