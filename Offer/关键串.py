'''
Description: 求解关键串的子串个数
Author: Luminary
Date: 2021-09-02 20:58:31
LastEditTime: 2021-09-02 21:00:38
'''
'''
定义关键串：当且仅当该串出现次数最多的字符超过了字符总数的一半
输入一个字符串求解是关键串的子串个数
样例：
输入：'ccccb'
输出：14
'''
import collections
def Keystring(str):
    N = len(str)
    count = 0
    for i in range(N):
        for j in range(N - i):
            s = str[j:j+i+1]
            c = collections.Counter(s)
            temp_num = c.most_common(1)[0][1]
            if temp_num > len(s) / 2:
                count += 1
    return count

a = Keystring('ccccb')
print(a)