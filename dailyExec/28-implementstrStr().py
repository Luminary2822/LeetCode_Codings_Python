'''
Description: 寻找一个字符串在另一个字符串出现的起始位置
Author: Luminary
Date: 2021-04-20 10:25:27
LastEditTime: 2021-04-20 10:26:33
'''
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        # 双指针滑动窗口判断是否包含子串，包含则返回其起始位置
        if not needle:
            return 0
        left,right = 0, len(needle)
        # 因为在字符串截取时左闭右开，所以right可以取到len(haystack)
        while right <= len(haystack):
            if haystack[left:right] == needle:
                return left
            left += 1
            right +=1
        return -1