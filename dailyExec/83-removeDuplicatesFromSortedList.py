# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        cur = head
        # 当后续结点存在的时候
        while cur.next:
            # 因为已经是升序链表所以重复元素必相邻，出现重复元素执行删除操作
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            # 未出现指针继续后移
            else:
                cur = cur.next
        return head
