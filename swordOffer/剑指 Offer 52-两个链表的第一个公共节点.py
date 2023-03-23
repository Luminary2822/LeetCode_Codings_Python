'''
Description: 两个链表的第一个公共节点
Author: Luminary
Date: 2021-07-22 14:20:51
LastEditTime: 2021-07-22 14:21:20
'''
class Solution:
    def getIntersectionNode(self, headA, headB):
        # 双指针
        if headA is None or headB is None:
            return None
        left, right = headA, headB
        while left != right:
            left = left.next if left else headB
            right = right.next if right else headA
        return left