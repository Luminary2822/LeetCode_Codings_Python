class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        # 第一种分奇数和偶数处理：O(n)复杂度【题目要求】
        bits = [0] * (num+1)
        for i in range(1, num+1):
            # 奇数二进制表示一定比前面那个偶数多一个 1
            if i % 2 == 1:
                bits[i] = bits[i-1] + 1
            # 偶数i二进制1的位数与 i/2的二进制1的位数相等（4:100,2:10,1的个数相同）
            # 偶数二进制末尾是0，除2相当于右移一位，1的个数没有变化
            else:
                bits[i] = bits[i//2]
        return bits
        # 第二种方法：直接转换二进制数1的个数，O(n*sizeof(integer))复杂度
        # res = []
        # for i in range(num+1):
        #     res.append(bin(i).count('1'))
        # return res

        # 为了补卡又写了一个版本，可能算法复杂度不符合要求但是可以通过
        """
        class Solution:
        def countBits(self, num: int) -> List[int]:
        # 好像复杂度不符合要求
        res = []
        for i in range(num+1):
            # 计算每个数中1的个数
            count = 0
            while i:
                # 将每个数对应二进制最低位1转换为0
                i &= i-1
                count += 1
            # 添加到结果数组中
            res.append(count)
        return res
        """