'''
Description: 旅行终点站
Author: Luminary
Date: 2021-10-01 13:21:23
LastEditTime: 2021-10-01 13:21:24
'''
class Solution:
    def destCity(self, paths):
        # 哈希表存储每条路径中的起点城市cityA
        cityA = {path[0] for path in paths}
        # 遍历路径中的cityB，如果在cityA的哈希表中不存在则说明该终点未做过其他路径的起点，即为旅行终点站
        for path in paths:
            if path[1] not in cityA:
                return path[1]
        
