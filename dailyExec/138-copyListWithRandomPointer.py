'''
Description: 复制带随机指针的链表
Author: Luminary
Date: 2021-07-23 13:06:36
LastEditTime: 2021-07-23 21:53:17
'''
"""
# Definition for a Node.
"""
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
        
class Solution:
    hashMap = dict()
    def copyRandomList(self, head) :
        # 浅拷贝： 返回地址一样的链表。
        # 深拷贝： 返回地址不一样，但关系一致的链表
        # 回溯法 + 哈希表：回溯遍历结点，哈希表存储旧结点到新结点的映射
        if head == None:
            return None
        if head in self.hashMap:
            return self.hashMap.get(head)
        newHead = Node(x = head.val)
        self.hashMap[head] = newHead
        newHead.next = self.copyRandomList(head.next)
        newHead.random = self.copyRandomList(head.random)
        return newHead