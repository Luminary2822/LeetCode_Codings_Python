class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        def backtrack(nums,path):
            if not nums:
                res.append(path)
                return
            for i in range(len(nums)):
                # 当前选择的元素和其他元素的全排列组合，不包含当前元素，已经选择的数字在当前要选择的数字中不能出现
                # [1,2,3]：先选出1，和[2,3]的全排列组合，累加到 path
                backtrack(nums[:i] + nums[i+1:],path + [nums[i]])
        backtrack(nums,[])
        return res

a = Solution()
print(a.permute([1,2,3]))
