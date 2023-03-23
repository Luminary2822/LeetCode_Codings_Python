class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # 从前往后遍历, 遇到一个数, 之前的所有集合添加上这个数, 组成新的子集，注意判断是否有重复子集
        res = [[]]
        # 去重先将原数组排序
        nums.sort()
        # 遍历数组中的数
        for i in range(len(nums)):
            size = len(res)
            # 遍历结果集内的子集
            for j in range(0,size):
                temp = list(res[j])
                # 将当前遍历到的数加入当前子集中
                temp.append(nums[i])
                # 判断当前结果集是否已经包含该子集，要求解集不能包含重复的子集
                # temp.sort()
                if temp not in res:
                    res.append(temp)
        return res

        # 回溯法：求包含重复元素数组的子集
        """
        def subsetsWithDup(self, nums):
            res = []
            # 去重需要先排序，便于重复元素前后比较
            nums.sort()
            self.dfs(nums, 0, res, [])
            return res
        def dfs(self, nums, index, res, path):
            # 这里要是判断path是否在res中出现，下面就不用判断两个相邻元素是否相同
            # 这里要是直接append不判断path是否出现，下面在path加入元素的时候就要判断相邻元素是否重复再决定是否加入
            if path not in res:
                res.append(path)
            for i in range(index, len(nums)):
                # 依次判断nums元素是否加入路径当中，当前元素与上一个元素重复则不加入其中
                # if i > index and nums[i] == nums[i-1]:
                #     continue
                self.dfs(nums, i + 1, res, path + [nums[i]])
        """