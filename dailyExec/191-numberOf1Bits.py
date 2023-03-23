class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 位运算
        count = 0 
        while(n):
            # 按位与：将二进制最低位的1变成0
            n &= (n-1)
            count += 1
        return count

        # 一句话解决一道题系列
        # return bin(n).count('1')