# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        # 思想：将链表每个节点向右移动 k 个位置，相当于把链表的后面 k % len  个节点移到链表的最前面。（len 为 链表长度）
        # 方法：用快慢指针找到链表的后面 k % len  个节点，slow指向这一段前驱，断开原链接设置新头结点，，fast与头结点相连将这段置于前端
        # 特判
        if not head or not head.next:
           return head
        cur = head
        len_listNode = 0
        # 求链表长度
        while cur:
            len_listNode += 1
            cur = cur.next
        # 长度取模
        k %= len_listNode
        if k == 0:
            return head
        # 设置快慢指针，慢指针和快指针相距 K，快指针指向 k+1
        slow, fast = head,head
        for _ in range(k):
            fast = fast.next
        # 快指针指到链表最后一个结点，慢指针slow指向链表倒数第 k+1 结点
        while fast.next:
            fast = fast.next
            slow = slow.next
        # 新链表的开头即为倒数第 k 个结点
        new_head = slow.next
        # 将倒数第 k+1 结点与 倒数第k个结点断开
        slow.next = None
        # 让链表最后一个结点指向头结点
        fast.next = head
        return new_head
        
