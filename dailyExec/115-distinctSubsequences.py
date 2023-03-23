#  s = "babgbag", t = "bag"                      
# 第一行, 字符串s都包含空字符串, 所以为1        
# 第一列, 空字符串不包含其他非空字符串, 所以为0 
#      ''  b  a  b  g  b  a  g                   
#  ''   1  1  1  1  1  1  1  1                   
#  b    0  1  1  2  2  3  3  3                   
#  a    0  0  1  1  1  1  4  4                   
#  g    0  0  0  0  1  1  1  5                   

class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        # 状态定义：dp[i][j] = s[:j]的子序列中 t[:i]出现的次数。
        # 状态转移方程：dp[i][j] = dp[i][j-1] + dp[i-1][j-1]/dp[i][j] = dp[i][j-1] 
        # t[i-1] 与 s[j-1]比较是否相等：不用s[j-1]这个字符或者用s[j-1]这个字符
        m = len(s)
        n = len(t)
        # 构建n+1行m+1列的状态转移数组，加入一行一列空串
        # dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)] 
        dp = [[0]*(m+1) for _ in range(n+1)]
        # 初始化定义一下dp数组，空串是任何字符串的子串
        for j in range(m+1):
            dp[0][j] = 1
        for i in range(1, n+1):
            for j in range(1, m+1):
                if t[i-1] == s[j-1]:        # j位置的状态主要取决于j-1的位置
                    # 两种情况：用s[j-1]这个字符（只考虑i-1和j-1的情况，当前字符已经固定） + 不用当前s[j-1]字符，行不变（t不变）列减1排除当前字符
                    dp[i][j] = dp[i-1][j-1] + dp[i][j-1]
                else:
                    # 不相等的情况直接不用当前字符
                    dp[i][j] = dp[i][j-1]
        # 返回dp矩阵最后的值就是结果
        return dp[-1][-1]