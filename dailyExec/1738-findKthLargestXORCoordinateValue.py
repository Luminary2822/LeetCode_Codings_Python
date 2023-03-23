'''
Description: 找出第 K 大的异或坐标值
Author: Luminary
Date: 2021-05-19 20:12:59
LastEditTime: 2021-05-19 20:13:22
'''
class Solution(object):
    def kthLargestValue(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        m = len(matrix)
        n = len(matrix[0])
        res = []

        # 将首行和首列空出来赋予默认值 0，并使用接下来的 m 行和 n 列存储二维前缀和
        # pre[i][j]表示(i-1,j-1)的值，即从(0,0)到(i-1,j-1)异或和
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1,m+1):
            for j in range(1,n+1):
                # 计算前缀异或和，小于i等于j ^ 小于j等于i ^ 小于i又小于j，最后异或matrix当前元素
                pre[i][j] = pre[i-1][j] ^ pre[i][j-1] ^ pre[i-1][j-1] ^ matrix[i-1][j-1]
                res.append(pre[i][j])
        # 对结果列表降序排序，取第k-1表示第k个最大值返回
        res.sort(reverse=True)
        return res[k-1]