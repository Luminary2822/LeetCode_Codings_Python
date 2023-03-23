import collections
import math
class Solution(object):
    def numRabbits(self, answers):
        """
        :type answers: List[int]
        :rtype: int
        """
        # 记录报相同数字的兔子最少种类数, value为报相同数字key的兔子的个数
        res = 0
        count = collections.Counter(answers)
        for key,value in dict(count).items():
            # (key + 1) * 颜色种类数 math.ceil(value / (key + 1)) 即为报该数字的兔子的最少总数
            # ceil()向上取整函数，也可以表示成 (n + x) // (x + 1)，//地板除向下取整取比目标结果小的最大整数
            res += math.ceil(value / (key + 1)) * (key + 1)
        return res

