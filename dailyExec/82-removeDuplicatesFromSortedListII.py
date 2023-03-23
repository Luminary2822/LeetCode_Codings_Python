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
        # 一次遍历，重复的元素在链表中出现的位置是连续的
        # 建立哑结点指向head
        dummpy = ListNode(0,head)
        cur = dummpy
        # 从第一个结点开始遍历，当连续两个结点存在时判断是否值相同
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                # 记录下重复的值
                x = cur.next.val
                # 从出现重复值的位置开始遍历，后续等于x需要删除
                while cur.next and cur.next.val == x:
                    cur.next = cur.next.next
            else:
                cur = cur.next
        return dummpy.next



        # 就很生气：调了好半天超时了好家伙，但是感觉运行起来应该是对的
        # 利用栈存储链表结点
        """
        stack = []
        dummpy = ListNode(0, head)
        cur = dummpy
        if not head:
            return head
        while cur.next:
            # 栈顶结点值与当前结点值相同的时候出栈，记录当前结点是存在重复情况的结点
            while stack and stack[-1].val == cur.val:
                x = stack.pop().val
                cur = cur.next
                # 指针移动判断后续结点值是否与前面存在重复情况的结点值相等，相等则还是重复结点不入栈
                if cur and cur.val == x:
                    cur = cur.next
            # 当前结点已经不是重复结点时时再入栈
            stack.append(cur)
            cur = cur.next
        # 哑结点将栈内剩余元素串联起来
        curr = dummpy
        for num in stack:
            curr.next = num
            curr = curr.next
        return dummpy.next
        """

            