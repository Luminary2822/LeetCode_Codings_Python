
# 判断是否是同构字符串
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # 第一种：哈希表解法
        hashMap = {}
        # zip方法打包成元组，建立两个字符串的键值对关系
        for i,j in zip(s, t):
            # s = "foo", t = "bar"，o键已经映射到r，不能再映射到a
            if i in hashMap and hashMap[i] != j:
                return False
            # s = "list", t = "fool"，s键不存在的情况下，o值已经被i映射，存在于字典中，不能再建立映射
            elif i not in hashMap and j in hashMap.values():
                return False
            # 正常情况就加入字典中
            hashMap[i] = j
        return True

        # 第二种：索引解法：同构字符串，每字符首次出现、最后出现、指定位出现索引始终相同
        """
        # 从字符串中找出某个子字符串第一个匹配项的索引位置
        return [s.index(i) for i in s] == [t.index(i) for i in t]
        """
     