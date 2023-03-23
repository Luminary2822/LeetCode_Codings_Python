'''
Description: 移除链表元素
Author: Luminary
Date: 2021-06-05 10:49:27
LastEditTime: 2021-06-05 10:59:21
'''
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        # 创建新结点作为链表头结点，方便将原head节点当成普通结点操作，可以进行删除
        new_head = ListNode(-1)
        # 新头结点指向原头结点
        new_head.next = head 
        # 记录新头结点，从.next开始遍历
        node = new_head

        # 开始遍历寻找删除值
        while node.next:
            if node.next.val == val:
                node.next = node.next.next
            else:
                node = node.next
        
        return new_head.next