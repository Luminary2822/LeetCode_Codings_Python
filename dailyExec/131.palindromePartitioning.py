class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        # 是否为回文串的判断函数
        self.isPalindrome = lambda s:s == s[::-1]
        res = []
        self.backtrack(s,res,[])
        return res
    # 回溯函数：寻找到达结束条件的所有可能路径
    def backtrack(self, s, res, path):
        # 未探索区域满足结束条件
        if not s:
            res.append(path)
            return
        # 切片操作[:i]实际为[0,i-1]，所以 i 要遍历到 len(s) + 1
        # 调试了一下，每次执行到这个循环的时候 path就为空了。
        for i in range(1, len(s) + 1):
            # 当前选择符合回文串条件
            if self.isPalindrome(s[:i]):
                # 递归回溯产生新数组
                self.backtrack(s[i:], res, path + [s[:i]])

a = Solution()
print(a.partition("aab"))