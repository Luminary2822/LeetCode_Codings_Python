'''
Description: 排序链表
Author: Luminary
Date: 2021-06-18 20:33:08
LastEditTime: 2021-06-19 16:28:49
'''
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 牺牲空间复杂度
        # 用列表存储链表结点的值 -> 列表排序 -> 将列表值输出增加结点构成新链表
        # node_val = []
        # while head:
        #     node_val.append(head.val)
        #     head = head.next
        # node_val.sort()
        # new_head = ListNode(-1)
        # cur = new_head
        # for n in node_val:
        #     node = ListNode(-1)
        #     node.val = n
        #     cur.next = node
        #     cur = cur.next
        # return new_head.next

        # 递归归并排序
        # 通过快慢指针找到链表中点-断链-合并
        # 1.通过快慢指针找到链表中点
        if head is None or head.next is None:
            return head
        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # 2. 断链 + 递归排序
        rightHead = slow.next
        slow.next = None
        left = self.sortList(head)
        right = self.sortList(rightHead)
        # 3. 迭代合并
        return self.mergeTwoLists(left, right)


    # 3.迭代合并
    def mergeTwoLists(self, l1,l2) :
        newHead = ListNode(-1)
        cur = newHead
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 if l1 else l2
        return newHead.next