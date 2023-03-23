'''
Description: 员工的重要性
Author: Luminary
Date: 2021-05-01 20:07:53
LastEditTime: 2021-05-01 20:08:35
'''
"""
# Definition for Employee.
class Employee(object):
    def __init__(self, id, importance, subordinates):
    	#################
        :type id: int
        :type importance: int
        :type subordinates: List[int]
        #################
        self.id = id
        self.importance = importance
        self.subordinates = subordinates
"""

class Solution(object):
    def getImportance(self, employees, id):
        """
        :type employees: List[Employee]
        :type id: int
        :rtype: int
        """
        # 先用哈希表存储员工ID及其员工信息[id:e]
        hashTable = dict()
        for e in employees:
            hashTable[e.id] = e
        # 深度优先遍历，对指定ID员工返回其重要度加遍历下属的重要度之和
        def dfs(ID):
            e = hashTable[ID]
            return e.importance + sum(dfs(subID) for subID in e.subordinates)
        # 初始对指定id进行dfs
        return dfs(id)