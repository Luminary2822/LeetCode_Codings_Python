class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 慢指针表示处理出的数组的长度，快指针表示已经检查过的数组的长度
        slow = 0
        for fast in range(len(nums)):
            # 检查上个应该被保留的元素nums[slow−2]是否和当前待检查元素 nums[fast]相同，相等时当前待检察元素不能被保留
            if slow < 2 or nums[slow-2] != nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
        return slow