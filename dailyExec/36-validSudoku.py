'''
Description: 有效的数独
Author: Luminary
Date: 2021-09-17 20:22:59
LastEditTime: 2021-09-17 20:23:00
'''
class Solution:
    def isValidSudoku(self, board) :
        # 遍历棋盘记录行、列、方块中的数字，用集合存储每行、每列、每个box内部的数字
        # 注意box和i，j下标序号的转换:pos = (i//3)*3 + j//3

        row = [set() for i in range(9)]
        column = [set() for i in range(9)]
        box = [set() for i in range(9)]

        for i in range(9):
            for j in range(9):
                item = board[i][j]
                pos = (i // 3) * 3 + j // 3
                if item != '.':
                # 判断该数字是否已经在该行该列以及该box块是否出现过，如果没有出现过则添加到set中
                    if item not in row[i] and item not in column[j] and item not in box[pos]:
                        row[i].add(item)
                        column[j].add(item)
                        box[pos].add(item)
                # 如果出现过就返回False
                    else:
                        return False
        return True