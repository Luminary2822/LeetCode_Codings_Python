
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # 迭代方法
        new_head = ListNode(-1)
        cur = new_head
        # 依次比较两个链表的结点值，将小的值连接到新链表后
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        # l1或l2剩余部分链接到新链表后面
        cur.next = l1 if l1 else l2
        return new_head.next

        # 递归方法
        """
        # 空链表判断
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        # 判断哪个链表的头结点值更小，递归地决定下一个添加到结果里的节点。如果两个链表有一个为空，递归结束
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
        """
