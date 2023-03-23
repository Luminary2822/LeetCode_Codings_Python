# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # 结果列表设置头结点
        res = ListNode(-1)
        # 设置当前指针
        cur = res
        # 向前进位
        carry = 0
        # 注意如果遍历完进位还存在时要记录进位
        while(l1 or l2  or carry):
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            # 逐位相加，carry为上一个的进位
            s = carry + x + y
            # 获取当前进位
            carry = s // 10
            # 记录当前的值
            cur.next = ListNode(s%10)
            # 移动指针
            cur = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return res.next
