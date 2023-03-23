class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 生成方式：在已经生成的丑数集合中乘以 [2, 3, 5] 而得到新的丑数
        if n < 0 :
            return 0
        # dp[i] 表示第 i - 1 个丑数（注dp[0]表示第一个丑数）
        dp = [1] * n
        # 三个指针表示 pi 的含义是有资格同 i相乘的最小丑数的位置
        index2, index3, index5 = 0, 0, 0
        for i in range(1, n):
            # 每次我们都分别比较有资格同2，3，5相乘的最小丑数，选择最小的那个作为下一个丑数
            dp[i] = min(2 * dp[index2], 3 * dp[index3], 5 * dp[index5])
            # 判断dp是由哪个相乘得到的，它失去了同 i 相乘的资格，把对应的指针相加
            if dp[i] == 2 * dp[index2]: index2 += 1
            if dp[i] == 3 * dp[index3]: index3 += 1
            if dp[i] == 5 * dp[index5]: index5 += 1
        return dp[n - 1]
