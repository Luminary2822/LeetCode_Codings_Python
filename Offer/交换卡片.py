'''
Description: 输出交换卡片的序号
Author: Luminary
Date: 2021-07-31 21:18:43
LastEditTime: 2021-09-02 21:00:53
'''
# 按从小到大排序好的卡片，有人交换了其中两张变成无序，请输出交换卡片的序号
# 输入[2,6,4,5,3]
# 输出[2,5],交换的是第二张和第五张
def find_differentNum(nums):
    res = []
    for i in range(len(nums)-1):
        # 找第一个比前一个小的数
        if nums[i] > nums[i+1]:
            res.append(i+1)
            break
    for i in range(len(nums)-1, -1, -1):
        # 找第一个比后一个大的数
        if nums[i-1] > nums[i]:
            res.append(i+1)
            break
    return res

a = find_differentNum([6,3,4,5,1])
print(a)

