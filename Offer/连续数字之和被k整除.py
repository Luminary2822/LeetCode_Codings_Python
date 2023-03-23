'''
Description: nums中有多少连续数字之和可以被key整除
Author: Luminary
Date: 2021-09-02 21:00:18
LastEditTime: 2021-09-02 21:01:22
'''
'''
nums中有多少连续数字之和可以被key整除
利用双指针，以左指针为起始，右指针在左指针后面依次进行组合累加，在组合过程中判断是否能被key整除并记录个数
样例：
输入：2，[4,5,1,-1,-2,-3]
输出：9
解释：以4开头连续数字和可以被2整除的有3个，以5开头有两个，以1开头有两个，以此类推累加起来。
'''
def kidNumber(key, nums):
    N = len(nums)
    left,right = 0,0
    sum = 0
    count = 0
    while left < N:
        while right < N:
            sum += nums[right]
            if sum % key == 0:
                count += 1
            right += 1
        left += 1
        sum = 0
        right = left
    return count

res = kidNumber(2, [4,5,1,-1,-2,-3])
print(res)