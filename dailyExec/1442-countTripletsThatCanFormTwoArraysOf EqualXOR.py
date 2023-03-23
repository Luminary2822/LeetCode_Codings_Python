'''
Description: 
Author: Luminary
Date: 2021-05-18 21:06:55
LastEditTime: 2021-05-18 21:11:32
'''
class Solution(object):
    def countTriplets(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        # [i,k]如果满足条件即累异或和应为0，因为由j分开两个相同数异或为0
        N = len(arr)
        ans = 0

        # i不能取到最后一位
        for i in range(N-1):
            # 计算[i,k]的累异或
            sum = 0
            for k in range(i,N):
                sum ^= arr[k]
                # 如果累异或为零，说明[i,k]除了i以外均可以当做j来分割
                if sum == 0:
                    # j不能取i可以取到k，所以是k-i
                    ans += (k - i)
        return ans