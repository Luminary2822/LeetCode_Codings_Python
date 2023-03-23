'''
Description: 链表中倒数第k个节点
Author: Luminary
Date: 2021-09-02 16:55:31
LastEditTime: 2021-09-02 16:57:25
'''
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getKthFromEnd(self, head, k):
        # 特判
        if not head or not head.next:
            return head
        # 快慢指针初始化：距离为k，slow指向第一个结点，fast指向第k+1个结点
        slow, fast = head, head
        for _ in range(k):
            fast = fast.next
        # 同时向前走：当fast指向空时，slow指向第k个结点
        while fast:
            fast = fast.next
            slow = slow.next
        return slow