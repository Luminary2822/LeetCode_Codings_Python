# 子集
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # 从前往后遍历, 遇到一个数, 之前的所有集合添加上这个数, 组成新的子集.
        res = [[]]
        # 从前往后遍历所有数
        for i in range(len(nums)):
            # 遍历之前的所有集合
            size = len(res)
            for j in range(0,size):
                temp = list(res[j])
                # 之前的每个集合都加上新的数
                temp.append(nums[i])
                # 组成新的集合再加入结果集
                res.append(temp)
        return res
        # 现在会回溯了，补充一款回溯写法：
        """
        def subsets(self, nums):
            res = []
            self.dfs(nums, 0, res, [])
            return res
        def dfs(self, nums, index, res, path):
            res.append(path)
            # 在nums后续数字中依次选择加入到路径当中
            for i in range(index, len(nums)):
                self.dfs(nums, i + 1, res, path + [nums[i]])
        """