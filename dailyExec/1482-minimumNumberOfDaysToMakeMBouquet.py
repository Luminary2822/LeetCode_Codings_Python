'''
Description: 制作 m 束花所需的最少天数
Author: Luminary
Date: 2021-05-09 15:31:06
LastEditTime: 2021-05-09 15:31:45
'''
class Solution(object):
    def minDays(self, bloomDay, m, k):
        """
        :type bloomDay: List[int]
        :type m: int
        :type k: int
        :rtype: int
        """
                # 对天数进行二分查找求取最小值
        left = min(bloomDay)
        right = max(bloomDay)

        # 特殊情况:一共需要m*k朵花，花园中花朵数量如果小于该值则不能摘到返回-1
        if m*k > len(bloomDay):
            return -1
        
        # 二分法查找可以制作成花束需要的最少天数
        while left < right:
            mid = (left + right) // 2
            # mid天可以制作成花束，则移动右指针判断更小的天数能否可以
            if self.checkDays(bloomDay, mid, m, k):
                right = mid
            # mid天不可以则移动左指针判断更大的天数
            else:
                left = mid + 1
        return left
                
    # 检查day天能否制作出m束花
    def checkDays(self, bloomDay, day, m, k):
        # 记录花朵数量
        flower = 0
        # 记录花束数量
        bouquet = 0
        for flower_day in bloomDay:
            # 必须是连续的花朵
            if flower_day <= day:
                flower += 1
                # 每满足一次连续的 k 朵花, 就可以制作一束花，制作完成flower归为0
                if flower == k:
                    bouquet += 1
                    flower = 0
            # 不连续则置为0
            else:
                flower = 0
            # 判断花束是否满足m束了，满足则跳出循环
            if bouquet >= m:
                break
        # 返回day天是否可以制作出m束花，满足返回Ture，否则返回False
        return bouquet >= m