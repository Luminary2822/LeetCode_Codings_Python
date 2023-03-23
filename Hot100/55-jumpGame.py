'''
Description: 跳跃游戏
Author: Luminary
Date: 2021-09-02 22:07:00
LastEditTime: 2021-09-02 22:07:01
'''
class Solution:
    def canJump(self, nums):
        # 贪心
        N = len(nums)
        far_distance = 0 
        for i in range(N):
            # 当前位置可以达到
            if i <= far_distance:
                # 更新维护最远可到达的位置far_distance
                far_distance = max(far_distance, i + nums[i])
                # 判断能否到达数组最后一个位置
                if far_distance >= N - 1:
                    return True
        return False