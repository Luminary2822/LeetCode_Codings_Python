'''
Description: 
Author: Luminary
Date: 2021-04-12 17:09:57
LastEditTime: 2021-04-12 22:38:44
'''
class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        # nums列表比较两个数不同的拼接顺序的结果，进而决定它们在结果中的排列顺序
        res = ''
        # 两两数字转变成字符串前后拼接比较大小，按照从前向后拼接结果较大排序
        for i in range(len(nums)-1):
            for j in range(i + 1, len(nums)):
                if str(nums[i]) + str(nums[j]) < str(nums[j]) + str(nums[i]):
                    nums[i], nums[j] = nums[j], nums[i]
        # 按照要求已经排好序的nums需要将里面元素依次转变字符串然后连接起来
        for x in nums:
            res += str(x)
        # 特殊情况判定，如果nums列表是多个0，转成字符串也是多个‘0’，但是实际要求只要一个‘0’
        # 所以判断最终结果字符串的首字符是否为0，因为如果nums中有不为0的字符一定排在0的前面
        if res[0] == '0':
            res = '0'
        return res