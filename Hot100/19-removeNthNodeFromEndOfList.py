class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
       # 设置哑结点是便于删除第一个元素
        dummpy = ListNode(0,head)
        slow,fast = dummpy,head
        # fast指向第n+1个结点
        for _ in range(n):
            fast = fast.next
        # fast指向最后为空时，slow指向倒数第n+1个结点（slow从dummpy开始，直到指向待删除结点前驱）
        while fast:
            fast = fast.next
            slow = slow.next
        # slow.next就是第n个结点
        slow.next = slow.next.next
        return dummpy.next