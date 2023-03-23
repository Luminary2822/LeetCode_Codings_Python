'''
Description: 单词搜索
Author: Luminary
Date: 2021-07-20 20:20:11
LastEditTime: 2021-07-20 20:23:30
'''
class Solution:
    def exist(self, board, word) :
        #深搜解法
        if not board:
            return False
        #搜索可能的起始位置
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board,word,i,j):
                    return True
        return False

    # 深度搜索
    def dfs(self,board,word,i,j):
        # dfs终止条件
        if len(word) == 0:
            return True
        # 边界终止条件
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[0]:
            return False
        # 不能搜索到之前的位置，设该位置为None
        tmp,board[i][j] = board[i][j],None
        # 向上下左右四个方向搜索
        res = self.dfs(board,word[1:],i-1,j) or self.dfs(board,word[1:],i+1,j) or self.dfs(board,word[1:],i,j-1) or self.dfs(board,word[1:],i,j+1)
        board[i][j] = tmp
        return res

