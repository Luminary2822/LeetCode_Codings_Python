# 字符串排列：滑动窗口+字典
import collections
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        # 统计s1的字符个数
        counter1 = collections.Counter(s1)
        # 定义滑动窗口的范围
        left = 0
        right = len(s1) - 1
        # 统计窗口s2[left, right - 1]内的元素出现的次数
        counter2 = collections.Counter(s2[0:right])
        while right < len(s2):
            # right位置元素加入s2中
            counter2[s2[right]] += 1
            # 如果滑动窗口内各个元素出现的次数跟 s1 的元素出现次数完全一致，返回 True
            if counter1 == counter2:
                return True
            # 窗口继续向右移动，把当前 left 位置的元素出现次数 - 1
            counter2[s2[left]] -= 1
            # 如果当前 left 位置的元素出现次数为 0， 需要从字典中删除，
            # 否则这个出现次数为 0 的元素会影响两 counter 之间的比较
            if counter2[s2[left]] == 0:
                del counter2[s2[left]]
            # 窗口继续向右移动
            left += 1
            right += 1
        return False
            
