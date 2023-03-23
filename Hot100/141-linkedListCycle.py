'''
Description: 环形链表
Author: Luminary
Date: 2021-06-20 19:09:41
LastEditTime: 2021-06-20 19:10:02
'''
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        # 快慢指针判断是否存在环，如果快慢指针相遇则存在环
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False