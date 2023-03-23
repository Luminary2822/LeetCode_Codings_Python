'''
Description: 变位词组
Author: Luminary
Date: 2021-07-18 13:56:32
LastEditTime: 2021-07-18 13:56:56
'''
class Solution:
    def groupAnagrams(self, strs) :
        # 建立映射关系【排序后的字符串作为键，原字母相同排列不同的字符串作为值】
        hashtable = dict()
        for str in strs:
            # sort完的str是list，需要转换成字符串形式作为key值存储到哈希表中
            str_key = ''.join(sorted(str))
            # 将排序后的str作为key，原str作为value存储，注意是[str]
            if str_key not in hashtable:
                hashtable[str_key] = [str]
            else:
            # 当前key存在的话，直接存储到对应key的value值下面
                hashtable[str_key].append(str)
        # 遍历哈希表中所有值即为结果
        return [value for value in hashtable.values()]
        