# 贪心：两个哈希表，分来用来每个数字对应的剩余次数，和存储数组中每个数字作为结尾的子序列的数量
import collections
class Solution(object):
    def isPossible(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
    # 创建键值对，一为nums所有数字和其对应的个数，二为空的
        countMap = collections.Counter(nums)
        endMap = collections.Counter()

        for x in nums:
            count = countMap[x]
            count1 = countMap.get(x+1,0)
            count2 = countMap.get(x+2,0)
            # get方法获取x-1键的值，没有的话默认值返回0
            preEndCount = endMap.get(x-1,0)
            if (count > 0):
                if preEndCount > 0:
                    countMap[x] -= 1
                    endMap[x-1] = preEndCount - 1
                    endMap[x] += 1
                else:
                    if (count1 > 0 and count2 > 0):
                        countMap[x] -= 1
                        countMap[x+1] -= 1
                        countMap[x+2] -= 1
                        endMap[x+2] += 1
                    else:
                        return False
        return True

a = Solution()
print(a.isPossible([1,2,3,3,4,4,5,5])) 