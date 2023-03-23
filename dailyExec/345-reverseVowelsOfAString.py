'''
Description: 反转字符串中的元音字母
Author: Luminary
Date: 2021-09-01 20:46:33
LastEditTime: 2021-09-01 20:47:03
'''
class Solution:
    def reverseVowels(self, s):
        # 双指针，注意字符串先转列表才能做交换
        s = list(s)
        left, right = 0 , len(s) - 1
        vowel_char = ['a', 'i', 'e', 'o', 'u', 'A', 'I', 'E', 'O', 'U']
        # 判断左右指针所到位置是否为元音字母，如果是就交换元音
        while left < right:
            while left < right and s[left] not in vowel_char:
                left += 1
            while left < right and s[right] not in vowel_char:
                right -= 1
            s[left], s[right] = s[right], s[left]
            # 交换完之后左右指针 
            left += 1
            right -= 1
        # 列表再转字符串
        return ''.join(s)