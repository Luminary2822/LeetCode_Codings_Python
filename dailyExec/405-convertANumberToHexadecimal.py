'''
Description: 数字转换为十六进制（有负数）
Author: Luminary
Date: 2021-10-02 14:09:43
LastEditTime: 2021-10-02 14:09:44
'''
class Solution:
    def toHex(self, num):
        # 32位2进制数，转换成16进制 -> 4个一组，一共八组
        base = "0123456789abcdef"
        res = []
        for _ in range(8):
            res.append(num%16)
            num //= 16
            if not num:
                break
        return "".join(base[n] for n in res[::-1])
