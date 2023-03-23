class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # 双指针
        # 每次交换，都是将左指针的零与右指针的非零数交换，且非零数的相对顺序并未改变。
        left,right = 0,0
        while right < len(nums):
            # 利用右指针移动判断当前元素是否为0，非0的话与左指针元素进行交换，然后移动左右指针
            if nums[right] != 0:
                nums[left],nums[right] = nums[right],nums[left]
                left += 1
            right += 1
        
        # 只用一行代码搞定过的天秀，key = bool只对非零元素和零元素排序，非零元素不排序
        # 降序：将非零元素排在前面，0排在后面，不影响非零元素的相对位置
        # nums.sort(key=bool, reverse=True)