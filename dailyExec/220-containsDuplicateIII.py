'''
Description: 存在重复元素III（桶排序）
Author: Luminary
Date: 2021-04-18 18:52:32
LastEditTime: 2021-04-18 19:40:13
'''
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        # 桶排序，地板除t+1遍历数组得到应放入桶的序号，利用哈希表存储桶和里面对应元素
        # 遍历元素得到桶id检查当前桶是否已被创建，如果被创建说明里面有一元素，当前元素和其满足条件返回True
        # 如果没有被创建则创建放入元素，加入字典关系
        # 桶内只有一元素时判断相邻桶内对应位置元素是否会出现绝对值小于等于t的情况
        # 当前遍历元素位置超过k的时候，就需要把i-k的数从桶里删除，因为不再考虑他们了
        all_buckets = {}    # 桶的集合
        bucket_size = t + 1
        for i in range(len(nums)):
            # i < k 的情况
            bucket_id = nums[i] // bucket_size
            # 桶已存在说明已有元素，当前元素也属于该桶，二者绝对值之差必然小于等于t
            if bucket_id in all_buckets:
                return True
            # 不存在即创建
            all_buckets[bucket_id] = nums[i]
            # 判断相邻桶内元素
            if (bucket_id - 1) in all_buckets and abs(all_buckets[bucket_id-1] - nums[i]) <= t:
                return True
            if (bucket_id + 1) in all_buckets and abs(all_buckets[bucket_id+1] - nums[i]) <= t:
                return True
            # i >= 没有返回True接下来i马上进入i+1，当i变为i+1的时候，i-k位置的桶已经不再考虑，所以删除
            if i >= k:
                all_buckets.pop(nums[i - k] // bucket_size)
        return False