'''
Description: 验证二叉搜索树
Author: Luminary
Date: 2021-07-18 22:01:33
LastEditTime: 2021-07-18 22:02:30
'''
# 递归
class Solution:
    def isValidBST(self, root) :
        res = []
        # 把二叉搜索树按中序遍历写成list
        def buildalist(root):
            if not root: return  
            buildalist(root.left)  
            res.append(root.val)  
            buildalist(root.right) 
            return res  
        buildalist(root)
        return res == sorted(res) and len(set(res)) == len(res) 

# 迭代
class Solution:
    def isValidBST(self, root) :
        # 迭代法构建中序遍历，res存储中序遍历值
        stack = []
        res = []
        cur = root
        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        # 检查list里的数有没有重复元素，以及是否按从小到大排列
        return res == sorted(res) and len(set(res)) == len(res)
 