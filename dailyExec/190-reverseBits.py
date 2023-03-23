class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        # n & 1:得到的是将 n 转换为二进制后的最后一位
        # " << "：左移运算符，将指定二进制向左移动一位，低位补0
        # " >> "：右移运算符，将制定二进制向右移动一位。
        #" | "：按位或运算符：只要对应的二个二进位有一个为1时，结果位就为1
        res = 0
        for _ in range(32):
            # res左移末尾为 0 ，取n的末尾二进制位，为1则按位或运算res末尾位为1，为0则按位或运算res末尾位为0，res前面其他位不变。
            # 相当于将n的末尾二进制位加到res末尾
            res = (res << 1) | (n & 1)
            # n继续右移
            n >>= 1
        return res