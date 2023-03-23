class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        # 按照第一维降序第二维升序排序
        people.sort(key= lambda x:(-x[0],x[1]))
        res = []
        # 判断当前结果列表中排在前面的人数len(res) 与 当前遍历people比他高应该排在他前面的人数p[1]
        for p in people:
            if len(res) <= p[1]:
                res.append(p)
            # 当前排在它前面的len(res)比应该排在前面的p[1]多，则将它插入到应该排在前面人数的位置p[1]
            else:
                res.insert(p[1], p)
        return res



a = Solution()
print(a.reconstructQueue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]))