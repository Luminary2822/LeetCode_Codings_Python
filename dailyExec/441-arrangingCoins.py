'''
Description: 排列硬币
Author: Luminary
Date: 2021-10-10 17:32:11
LastEditTime: 2021-10-10 17:32:12
'''
class Solution:
    def arrangeCoins(self, n):
        # 等差数列计算公式可得：第 0 ~ i 行总的硬币数目有 (i + 1) * i / 2
        # 二分查找满足(i + 1) * i / 2  <= n的最大数字i

        # 二分查找双闭区间
        left, right = 0, n
        while left <= right:
            mid = (left + right) // 2
            if mid * (mid + 1)  <= 2 * n:
                left = mid + 1
            else:
                right = mid - 1
        # 最后left会在mid基础上多加一步，所以要减一
        return left - 1