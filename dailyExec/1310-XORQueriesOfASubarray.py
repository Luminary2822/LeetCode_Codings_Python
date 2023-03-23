'''
Description: 子数组异或查询
Author: Luminary
Date: 2021-05-12 20:11:54
LastEditTime: 2021-05-12 20:12:20
'''
class Solution(object):
    def xorQueries(self, arr, queries):
        """
        :type arr: List[int]
        :type queries: List[List[int]]
        :rtype: List[int]
        """
        # 构造前缀异或累计数组mp，mp[i] 是所有 0 到 i 元素的与或的结果
        N = len(arr)
        # mp[0] = 0^arr[0] = arr[0]
        cur = 0 
        mp = []
        for i in range(N):
            # 累异或结果
            cur ^= arr[i]
            mp.append(cur)
        
        res = []
        # 遍历queries中每组元素的两个位置
        for l, r in queries:
            # 如果首位置为0，直接返回末位置前缀和
            if l == 0:
                res.append(mp[r])
            # 首位置-1的异或前缀 异或 末位置异或前缀结果，相同异或为0消除
            # 例如[1,3,4,8]，求(1,2)即为mp[0] ^ mp[2]，即1 ^ 1^3^4 = 3^4
            else:
                res.append(mp[l-1]^mp[r])
        return res