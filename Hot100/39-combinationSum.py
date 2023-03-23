class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        # 判空
        if not candidates:
            return []
        res = []
        # 先排序，便于挑选过程中遍历到元素累加已经大于target即return无需再继续
        candidates.sort()
        # 回溯方法：注意剪枝
        def backtrace(i,temp_sum,temp_list):
            # 当前累加和满足退出条件，即把当前路径加入结果集
            if temp_sum == target:
                res.append(temp_list)
                return
            # 剪枝：当前累加和大于目标无需再继续回溯
            if temp_sum > target:
                return
            # 选择未探索区域进行新的回溯，累加和累加，累加列表累加
            for j in range(i,len(candidates)):
                backtrace(j,temp_sum+candidates[j],temp_list+[candidates[j]])
        # 入口
        backtrace(0,0,[])
        return res