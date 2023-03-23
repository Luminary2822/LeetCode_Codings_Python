'''
Description: H指数
Author: Luminary
Date: 2021-07-11 13:59:49
LastEditTime: 2021-07-11 14:00:16
'''
class Solution:
    def hIndex(self, citations):
        # 维基百科：一个人在其所有学术文章中有N篇论文分别被引用了至少N次，他的H指数就是N
        # 逆序排序
        citations.sort(reverse = True)
        N = len(citations)
        res = 0
        # 从高到低判断当前引用数citations[i]是否大于等于(至少)当前论文数(i+1)，如果是的话则i+1记录为一个H指数
        for i in range(N):
            if citations[i] >= i + 1:
                res = max(res, i + 1)
            else:
                break
        return res

