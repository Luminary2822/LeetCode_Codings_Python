# 汉明距离
class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        # 异或：相同为0,不同为1，看结果中1的个数
        n = x ^ y
        cnt = 0
        # n不为0的情况下可以计数出有多少1
        while n:
            # 按位与的作用：将n的二进制表示中最低位的1改成0
            n &= (n-1)
            cnt += 1
        return cnt

a = Solution()
print(a.hammingDistance(1,4))


