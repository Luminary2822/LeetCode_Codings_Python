'''
Description: 环形链表II
Author: Luminary
Date: 2021-09-04 13:09:03
LastEditTime: 2021-09-04 13:09:04
'''
class Solution:
    def detectCycle(self, head) :
        # 判断是否存在环，找到环相遇的结点
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                # 从头结点和相遇结点出发，各寻找环的入口结点
                p = head
                q = slow
                while p != q:
                    p = p.next
                    q = q.next
                return p
        return None