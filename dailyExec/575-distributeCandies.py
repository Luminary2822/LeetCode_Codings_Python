'''
Description: 分糖果
Author: Luminary
Date: 2021-11-01 11:48:58
LastEditTime: 2021-11-01 11:48:59
'''
class Solution:
    def distributeCandies(self, candyType):
        # 糖果数：len(candyType)
        # 糖果种类：len(set(candyType))
        # 妹妹分得一半糖果，且分到的糖果种类不超过总数
        return min(len(set(candyType)), len(candyType)//2)
