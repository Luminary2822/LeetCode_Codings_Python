'''
Description: 相交链表
Author: Luminary
Date: 2021-06-04 15:00:58
LastEditTime: 2021-06-05 10:19:00
'''

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if headA is None or headB is None:
            return None
        # 双指针
        left, right = headA, headB
        # 两个指针走相同的路消除长度差，可以同时到达终点
        while left != right:
            # 如果先走到链表尾部，移动指针到另一链表头结点
            left = left.next if left else headB
            right = right.next if right else headA
        return left