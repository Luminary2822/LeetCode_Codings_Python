class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        result = []
        # 从0开始遍历到numRows
        for i in range(numRows):
            # 每行生成对应个数全为1的列表
            now = [1] * (i+1)
            for n in range(1, i):
                # 当前位置等于前一行当前位置-1元素值+前一行当前位置值
                now[n] = pre[n-1] + pre[n]
            # 以列表嵌套的形式加入结果列表中
            result += [now]
            # 前一行的指向移动
            pre = now
        return result

a = Solution()
print(a.generate(5))