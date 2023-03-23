'''
Description: 解码异或后的数组
Author: Luminary
Date: 2021-05-06 19:27:48
LastEditTime: 2021-05-06 19:28:15
'''
class Solution(object):
    def decode(self, encoded, first):
        """
        :type encoded: List[int]
        :type first: int
        :rtype: List[int]
        """
        # 异或满足交换律
        #  c^b = a
        #  c^a = b
        res = [first]
        for x in encoded:
            # 先利用first解码出原数组第二个元素，再利用第二个异或encoded求解第三个，以此类推
            # res[-1]表示上一个解码的元素
            res.append(x^res[-1])
        return res