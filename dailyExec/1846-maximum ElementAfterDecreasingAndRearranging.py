'''
Description: 减小和重新排列数组后的最大元素
Author: Luminary
Date: 2021-07-16 12:57:00
LastEditTime: 2021-07-16 12:57:34
'''
class Solution:
    def maximumElementAfterDecrementingAndRearranging(self, arr):
        # 排序之后用贪心
        arr.sort()
        # 限定首位为1
        arr[0] = 1
        # 依次判断数组中的元素是否与上一位仅相差为1，
        for i in range(1, len(arr)):
            # 相差小于等于1不变，相差大于1，则将当前数改成仅比上一位数大1的数
            if (arr[i] - arr[i-1]) > 1:
                arr[i] = arr[i-1] + 1
        # 遍历结束后最后的数就是数组中最大的数
        return arr[-1]