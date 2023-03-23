import collections
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 哈希表存储每个元素出现的次数，形成元素-次数键值对
        counter = collections.Counter(nums)
        n = len(nums)
        # 遍历找到对应次数大于n/2的值，返回其对应的键
        for num in nums:
            if counter[num] > int(n/2):
                return num
        # 官方题解比较厉害的一句话:其实就是在找数组的众数，出现次数最多的数
        # 把 counter的 keys 作为目标, key 定义成 counter 的 get 方法
        # return max(counter.keys(),key = counter.get)

        # 第二种方法：排序，下标为n/2的数一定是众数
        """
        nums.sort()
        return nums[len(nums) // 2]
        """