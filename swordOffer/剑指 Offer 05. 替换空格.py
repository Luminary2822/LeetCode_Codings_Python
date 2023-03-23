'''
Description: 替换空格
Author: Luminary
Date: 2021-10-07 14:22:37
LastEditTime: 2021-10-07 14:22:37
'''
class Solution:
    def replaceSpace(self, s):
        # 扩充数组利用双指针法从后向前替换空格

        # 扩充数组到每个空格替换成"%20"之后的大小
        # 每碰到一个空格就多拓展两个格子，1 + 2 = 3个位置存’%20‘
        space_num = s.count(' ')
        res = list(s)
        res.extend([' '] * space_num * 2)

        # 从原字符串的末尾和扩展后字符串的末尾，双指针向前遍历
        left = len(s) - 1
        right = len(res) - 1

        while left >= 0:
            # 左指针指向不是空格，则复制字符到右指针
            if res[left] != ' ':
                res[right] = res[left]
                right -= 1
            # 左指针指向的是空格，右指针向前三个格填充%20
            else:
                res[right - 2: right + 1] = '%20'
                right -= 3 
            left -= 1
        return "".join(res)

            

