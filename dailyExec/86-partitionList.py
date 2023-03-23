# 分隔链表
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        # 两个链表分别存储比x小,与x相等或者比x大的元素，再将两个链表连接起来
        # 设置两个链表的值为0的头结点，less和more是移动的指针
        less_head = less = ListNode(0)
        more_head = more = ListNode(0)
        while head:
            if head.val < x:
                less.next = head
                less = less.next
            else:
                more.next = head
                more = more.next
            head = head.next
        # 设置最后结点指向为空
        more.next = None
        # 链表连接：more_head为自己初始化的结点值为0，要指向next才是真的值
        less.next = more_head.next
        # 返回连接后链表的结点，less_head为自己初始化的结点值为0
        return less_head.next