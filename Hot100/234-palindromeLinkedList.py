class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        # 第一种方法：利用栈判断牺牲空间复杂度，先将链表入栈再从表头和栈顶比较
        # 边界条件判断
        if not head and head.next:
            return True
        stack = []
        # 将所有元素入栈
        cur = head
        while cur:
            stack.append(cur.val)
            cur = cur.next
        # 表头元素与栈顶元素比较，相等则出栈继续向内判断
        cur = head
        while stack:
            if cur.val != stack.pop():
                return False
            else:
                cur = cur.next
        return True
        
        # 第二种方法：直接使用列表逆序判断
        """
        res = []
        while head:
            res.append(head.val)
            head = head.next
        return res == res[::-1]
        """
