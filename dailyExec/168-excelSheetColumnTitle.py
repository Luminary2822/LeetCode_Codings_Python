'''
Description: Excel表列名称
Author: Luminary
Date: 2021-07-31 13:52:55
LastEditTime: 2021-07-31 13:53:05
'''
class Solution:
    def convertToTitle(self, columnNumber) :
        res = ''
        while columnNumber:
            # A对应的序号是1不是0，所以列号要减1
            columnNumber -= 1
            # 这里不能用+=要用+，不然28应该是AB会输出BA，先计算出来的字符在后面
            res = chr(columnNumber % 26 + 65) + res
            columnNumber //= 26
        return res