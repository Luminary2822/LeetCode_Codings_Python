# 计数质数
# 本题采用素数筛法里面的埃及筛，质数的倍数是合数
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return 0
        cnt = 0
        # 设置一个都为false的数组
        flag = [ False ] * n
        for i in range(2, n):
            # 累计如果是false就是质数
            if flag[i] == False:
                cnt += 1
                # 从i*i开始筛选i的倍数设置为合数，小于i的已经筛选过了
                for j in range(i*i, n, i):
                    flag[j] = True
        return cnt
a = Solution()
print(a.countPrimes(10))

# 解释说明：
# 筛5的倍数时，我们从5的2倍开始筛，但是5 * 2会先被2 * 5筛去， 5 * 3会先被3 * 5会筛去，
# 5 * 4会先被2 * 10筛去，所以我们每一次只需要从i*i开始筛，因为(2，3,…,i - 1)倍已经被筛过了。

