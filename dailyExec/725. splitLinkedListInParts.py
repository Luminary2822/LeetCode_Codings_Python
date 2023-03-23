'''
Description: 分隔链表
Author: Luminary
Date: 2021-09-22 15:52:46
LastEditTime: 2021-09-22 15:52:46
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def splitListToParts(self, head, k):
        # 计算k个部分每部分最少多少结点，分完k部分多出的结点平分到前面去
        # 按照每部分结点数量，将每部分的头结点纳入res，切断后续链表
        
        #1. 特殊情况判断
        if not head:
            return [None for _ in range(k)]
        
        #2. 计算链表长度 
        n = 0
        p = head
        while p:
            n += 1
            p = p.next
        
        #3. 计算k部分每部分最少多少结点，以及多余出来多少结点
        part_len = n // k
        remain_len = n % k
        # 列表存储k个值，每个值为每部分的长度
        part = [part_len] * k
        # 将多出来的结点均分在之前的结点上
        for i in range(remain_len):
            part[i] += 1
        
        #4. 开始遍历添加每一部分的起始结点到res中【注意head是ListNode型，append进去就是小List形式】
        res = [head]
        for i,num in enumerate(part):
            # 按照part内存储的长度遍历每一部分，head最后指向下一部分的起始结点，pre指向该遍历部分的最后结点
            for _ in range(num):
                pre = head
                head = head.next
            # 每次是在每一部分循环开始前添加该部分的头结点head，并切断上一部分最后结点的next。
            if i != k-1:
                pre.next = None
                res.append(head)
        return res
        

