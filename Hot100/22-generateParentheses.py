class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        #  特殊况特判：
        if n <= 0:
            return []
        res = []

        # 回溯法深度优先遍历：括号生成问题抽象成一颗树结构：满二叉树遍历，再根据左右括号的数量剪枝
        def dfs(path, left, right):
            if left > n or right > left:
                return
            # 树的一条路径长度等于2*n的时候是有效组合（n对括号共有2*n个括号）
            if len(path) == 2*n:
                res.append(path)
                return
            dfs(path+'(', left+1, right)
            dfs(path+')', left, right+1)

        # 回溯起点
        dfs("",0, 0)
        return res