# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        # 设置单链表反转函数：给出头结点链表反转
        def reverseLinkNode(head):
            pre = None
            cur = head
            while cur:
                temp = cur.next
                cur.next = pre
                pre = cur
                cur = temp
        # 设置虚拟头结点以防头结点发生变化
        dummpy_node = ListNode(-1)
        dummpy_node.next = head
        pre = dummpy_node
        # 第一步：找到原链表中左边位置结点的前驱pre和右边位置结点right_node
        for _ in range(left-1):
            pre = pre.next
        right_node = pre
        for _ in range(right - left + 1):
            right_node = right_node.next
        
        # 第二步：将子链表切除出来，保留右节点的后继
        left_node = pre.next
        succ = right_node.next
        # 切断连接
        pre.next = None
        right_node.next = None
        # 反转子链表，反转之后的链表头结点right_node，尾结点为left_node
        reverseLinkNode(left_node)
        # 将反转好的子链表和前后连接起来
        pre.next = right_node
        left_node.next = succ

        return dummpy_node.next


        # 第二种方法：利用栈存储待翻转的子链表
        """
        p = head
        stack = []
        # 判断当前遍历位置是否在left和right内部
        loc = 1
        while p :
            # 将子链表区间内的值入栈
            if loc in [left,right]:
                stack.append(p.val)
            p = p.next   
            loc += 1
        # 二次遍历出栈替换链表原位置的值
        res = head
        loca = 1
        while res:
            if loca in [left,right]:
                res.val = stack.pop()
            res = res.next
            loca += 1
        return head
        """

        # 第三种方法：一次遍历头插法
        # 在需要反转的区间里，每遍历到一个节点，让这个新节点来到反转部分的起始位置
        """
        dummpy_node = ListNode(-1)
        dummpy_node.next = head
        pre = dummpy_node
        # pre:永远指向待翻转区域第一个结点的前驱
        for _ in range(left-1):
            pre = pre.next
        # 指向待翻转区域的第一个结点
        cur = pre.next
        for _ in range(right - left):
            # next永远指向 curr 的下一个节点，随着cur的变化而变化
            next = cur.next
            cur.next = next.next
            next.next = pre.next
            pre.next = next
        return dummpy_node.next
        """



        


        


        


        

