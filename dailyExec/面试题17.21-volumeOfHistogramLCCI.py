class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # 双指针，按行遍历，总体积减去柱子体积就是水的容量
        # 利用左右指针的下标差值计算出每一层体积
        Sum = sum(height)         # 柱子体积
        size = len(height)        # 区域长度
        left, right = 0, size - 1 # 双指针
        volumn, high = 0, 1       # 总体积和高度初始化
        while left <= right:
            # 当左右指针指向的区域高度小于high时，左右指针都向中间移动
            while(left <= right and height[left] < high):
                left += 1
            while(left <= right and height[right] < high):
                right -= 1
            # 直到指针指向大于等于high的时候，计算该层的体积，累加到volumn
            volumn += right - left + 1
            # 层数累加
            high += 1
        return volumn - Sum

        # 附加一下比较容易想到的暴力解法【简单解法】
        """
        # 对于每个位置，向左右找最高的木板；当前位置能放的水量是：左右两边最高木板的最低高度 - 当前高度
        res = 0
        # 第 0 个位置和 最后一个位置不能蓄水，所以不用计算
        for i in range(1, len(height) - 1):
            # 求右边的最高柱子
            rHeight = max(height[i + 1:])
            # 求左边最高柱子
            lHeight = max(height[:i])
            # 左右两边最高柱子的最小值 - 当前柱子的高度
            h = min(rHeight,lHeight) = height[i]
            # 如果能蓄水
            if h > 0:
                res += h
        return res
        """




