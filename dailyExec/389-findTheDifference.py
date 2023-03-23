import collections
class Solution(object):
# 第一种：利用Counter相减方法
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # 建立字符与个数的键值对
        num1 = collections.Counter(t)
        num2 = collections.Counter(s)
        # 获得差异字符的键值对
        diffCounter = num1 - num2
        # elements方法将键列出转成list获取差异字符
        diff = list(diffCounter.elements())[0]
        return diff
  
a = Solution() 
print(a.findTheDifference('dfgdfg','dfegdfg'))

#第二种：Counter+next迭代的方法
"""
return next((Counter(t)-Counter(s)).elements())
"""

#第三种：遍历 Counter里面的key
"""
def findTheDifference(self, s: str, t: str) -> str:
    dict_s = collections.Counter(list(s))
    dict_t = collections.Counter(list(t))
for key in dict_t.keys():
    # 这里的or是为了ae和aea这样的情况，t内新增加的差异与已有字符重复，key也在s内，但是值不同。
    if key not in dict_s.keys() or dict_t[key] != dict_s[key]:
        return key
"""

#第四种：位运算
"""
class Solution(object):
    def findTheDifference(self, s, t):
        res = 0    # 0^任何非0数都为它本身
        for a in s:
            res ^= ord(a)   # ord() 是将一个字符转化为ASCII值 
        for a in t:
            res ^= ord(a)
        return chr(res)     #将额外元素的ASCII转化为字符
"""


