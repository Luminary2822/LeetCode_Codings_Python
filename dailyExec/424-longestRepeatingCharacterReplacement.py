# 替换后最长重复字符：滑动窗口
import collections
class Solution(object):
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        # 设置左指针和结果长度
        left, res= 0, 0
        # 设置字典存储每个字符的出现次数
        cur = collections.defaultdict(int)
        # 遍历字符串中每个字符和下标位置
        for right, val in enumerate(s):
            # 存储出现次数
            cur[val] += 1
            # 窗口内可替换字母数量大于K的时候，移动左指针向右
            while right - left + 1 - max(cur.values()) > k:
                cur[s[left]] -= 1
                left += 1
            # 获得当前窗口最大值
            res = max(res, right - left + 1)
        return res



