'''
Description: 任务调度器
Author: Luminary
Date: 2021-06-27 12:07:02
LastEditTime: 2021-06-27 12:29:54
'''
from typing import List
import collections
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        # https://leetcode-cn.com/problems/task-scheduler/solution/tong-zi-by-popopop/
        # 桶思想：总排队时间 = (桶个数 - 1) * (n + 1) + 最后一桶的任务数 / 任务的总数
        # max[(最多的那个任务的任务数 - 1) * (间隔 + 1) + 任务数量等于最多的任务数的数量, len(tasks)]

        # 统计每个任务出现的频次
        count_tasks = collections.Counter(tasks)
        # 获得出现频次最高的任务的频次
        max_count = count_tasks.most_common(1)[0][1]
        # 计算频次最高的任务有几个
        last_size = list(count_tasks.values()).count(max_count)

        # 计算矩形面积：最后一轮不满单独计算
        res = (max_count - 1) * (n + 1) + last_size
        # 如任务总类和数量充足，无需等待，直接返回len(tasks)
        return max(res, len(tasks))