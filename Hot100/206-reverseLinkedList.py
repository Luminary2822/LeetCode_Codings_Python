# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # cur表示当前遍历到的节点，pre表示当前节点的前驱节点，需要中间变量temp保存当前节点的后驱节点
        pre = None
        cur = head
        while cur:
            # 先把当前节点的后驱节点保存以免丢失
            temp = cur.next
            # 反转指向前驱节点
            cur.next = pre
            # 将pre，cur往后移一位
            pre = cur
            cur = temp
        # 当cur == Null结束循环返回pre
        return pre