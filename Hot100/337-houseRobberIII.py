'''
Description: 打家劫舍III（二叉树形）
Author: Luminary
Date: 2021-04-15 18:07:27
LastEditTime: 2021-04-15 18:08:33
'''
class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # _rob方法表示以root为根节点的树中，返回抢劫根节点与不抢劫根节点可获得的最大值
        def _rob(root):
             # 两个0分别表示偷/不偷该节点可获得的最大值
            if not root: return 0, 0 
            # left和right均为二维数组，第一维表示偷该节点获取的金额，第二维表示不偷该节点获取的金额
            # 递归对于以root.left为根节点的树计算抢劫根节点和不抢劫根节点可获得的最大金额，right同理
            left = _rob(root.left)
            right = _rob(root.right)
            # 偷当前节点, 则左右子树都不能偷
            v1 = root.val + left[1] + right[1]
            # 不偷当前节点, 则取左右子树中最大的值
            v2 = max(left) + max(right)
            return v1, v2

        return max(_rob(root))