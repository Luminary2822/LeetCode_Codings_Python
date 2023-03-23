'''
Description: 不同的二叉搜索树
Author: Luminary
Date: 2021-05-12 20:33:06
LastEditTime: 2021-05-12 20:34:37
'''
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        # dp[i]表示由i个结点构成二叉搜索树的个数,空树也是一种，所以dp[0]=1
        dp = [0] * (n+1)
        dp[0] = 1

        # 节点个数从1到n的所有情况
        for i in range(1, n+1):
            # 每种根节点对应的二叉树数量并求和
            for j in range(i):
                # 每种根节点对应的二叉树数量为其左子树数量乘以右子树数量
                dp[i] += dp[j] * dp[i-j-1]
        
        return dp[n]