# 求滑动窗口的中位数
# 求a的中位数的python3代码：(a[(len(a)-1)//2] + a[len(a)//2]) / 2，
# 思路返回中间值的和的一半。（如果长度为单数，则求和两次）
class Solution(object):
    def medianSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[float]
        """
        # 第一种：直接阅读理解题意简单法，运行环境为python3
        res = []
        i = 0
        # k为奇数flag = true
        flag = True if k % 2 == 1 else False
        # 滑动窗口为i,i + k - 1
        while i + k - 1 < len(nums):
            # 在python3的环境下切片和copy都可以复制列表，tmp指向不同的对象
            # 切片和copy两种方法都可以实现得到两个指向不同对象独立的列表
            # tmp = nums[i:i+k].copy()
            tmp = nums[i:i+k]
            tmp.sort()
            # k为奇数中位数取中间值，为偶数取中间两个值的平均数
            if flag:
                res.append(tmp[k // 2])
            else:
                val = tmp[k // 2] + tmp[k // 2 - 1]
                res.append(val/2)
            i += 1
        return res

        # 第二种：数组+二分查找
        """
        # python排序模块
        import bisect
        # 求中位数方法，匿名函数lambda接受参数并返回表达式的值
        median = lambda a: (a[(len(a)-1)//2] + a[len(a)//2]) / 2
        # 维护数组a保存当前数组且有序
        a = sorted(nums[:k])
        res = [median(a)]
        # i表示删除值，j表示插入值，滑动窗口a中i删除，j插入在合适的位置，求中位数加入结果集
        # 注意：i的取值范围nums[:-k]去除后k个的元素的列表
        for i, j in zip(nums[:-k], nums[k:]):
            # bisect_left将数插入到正确的位置且不会影响之前的排序，left如果有重复数值插入在左边
            a.pop(bisect.bisect_left(a, i))
            a.insert(bisect.bisect_left(a, j), j)
            res.append(median(a))
        return res
        """
# a = Solution
# res = a.medianSlidingWindow(a, [1,3,-1,-3,5,3,6,7], 3)
# print(res)