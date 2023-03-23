'''
Description: 字母异位词分组
Author: Luminary
Date: 2021-04-18 20:45:48
LastEditTime: 2021-04-18 20:46:48
'''
import collections
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        # 建立哈希表，映射每一组字母异位词，所以构造为列表形式的dict
        hashtable = collections.defaultdict(list)
        for str in strs:
            # 将字符串排序后得到的字符串作为键值，因为字母异位词排序后的字符串一定相同
            key = "".join(sorted(str))
            # 将对应字母异位词作为值加入到key对应的值列表中
            hashtable[key].append(str)
        # 将所有values以list形式返回
        return list(hashtable.values())