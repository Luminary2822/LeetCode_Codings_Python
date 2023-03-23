'''
Description: Excel表列序号
Author: Luminary
Date: 2021-07-31 13:39:42
LastEditTime: 2021-07-31 13:41:37
'''
class Solution:
    def titleToNumber(self, columnTitle):
        # 找规律
        # 单字母为ord('x') - ord('A') + 1
        # 双字母先按照单字母求解第一个字母，结果为起始倍数*26加上第二个单字母
        ans = 0
        for item in columnTitle:
            ans = ans * 26 + ord(item) - ord('A') + 1
        return ans