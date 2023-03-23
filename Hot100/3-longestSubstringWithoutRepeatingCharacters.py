import collections
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 维护一个列表队列/集合，遇到重复元素则向右移动窗口直至将该元素移除，再添加新元素
        left, res = 0,0
        c = []
        # c = set()
        for right,val in enumerate(s):
            # 判断遍历到的字符是否已经存在于窗口中，存在则向右移动窗口，直到将该重复元素移除窗口
            while val in c:
                c.remove(s[left])
                left += 1
            c.append(val)
            # c.add(val)
            res = max(res, right - left + 1)
        return res
a = Solution()
print(a.lengthOfLongestSubstring('pwwkew'))