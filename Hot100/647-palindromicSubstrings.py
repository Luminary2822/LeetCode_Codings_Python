class Solution(object):
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # dp[j]表示从j到当前遍历的i的字符串是否为回文子串
        N = len(s)
        # 初始化状态数组为 1，cnt为 N，因为每个单词自身均为一个回文串
        dp = [1] * N
        cnt = N
        for i in range(N):
            for j in range(i):
                # 如果当前字符串首尾相等且子串是回文串，j到当前i的字符串才为回文串，并计数
                if s[j] == s[i] and dp[j+1] == 1:
                    dp[j] = 1
                    cnt += 1
                # 非回文串设置为0
                else:
                    dp[j] = 0
        return cnt

        # 第二种方法：以中心扩散判断
        """
        L = len(s)
        cnt = 0
        # 以某一个元素为中心的奇数长度的回文串的情况
        for center in range(L):
            left = right = center
            while left >= 0 and right < L and s[left] == s[right]:
                cnt += 1
                left -= 1
                right += 1
        # 以某对元素为中心的偶数长度的回文串的情况
        for left in range(L - 1):
            right = left + 1
            while left >= 0 and right < L and s[left] == s[right]:
                cnt += 1
                left -= 1
                right += 1
        return cnt
        """  