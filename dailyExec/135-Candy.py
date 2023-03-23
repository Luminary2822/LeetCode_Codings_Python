class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        length = len(ratings)
        # 初始化一个N维数组，数据都为1（因为每个孩子至少一颗）
        res = [1 for _ in range(length)]
        # 1.从左往右扫描，比较ratings[i]与ratings[i-1]的值，res[i]不满足则增加糖果至满足
        # 2.从右往左扫描，比较ratings[i]与ratings[i+1]的值，res[i]不满足则增加糖果至满足
        for i in range(1,length):
            if(ratings[i] > ratings[i-1] and res[i] <= res[i-1]):
                res[i] = res[i-1] + 1
        for i in range(length-2, -1, -1):
            if(ratings[i] > ratings[i+1] and res[i] <= res[i+1]):
                res[i] = res[i+1] + 1
        return sum(res)
