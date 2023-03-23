# 三个方法：
# isInteger() ，判断当前存储的对象是否为 int；
# getInteger() , 如果当前存储的元素是 int 型的，那么返回当前的结果 int，否则调用会失败；
# getList() ，如果当前存储的元素是 List<NestedInteger> 型的，那么返回该 List，否则调用会失败。

class NestedIterator(object):
    # 迭代：在出栈过程中判断是数字或是列表，列表直接展开入栈
    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        # 利用栈存储嵌套列表内的元素
        self.stack = []
        for i in range(len(nestedList)-1,-1,-1):
            self.stack.append(nestedList[i])

    def next(self):
        """
        :rtype: int
        """
        res = self.stack.pop()
        return res.getInteger()
    
    def hasNext(self):
        """
        :rtype: bool
        """
        while self.stack:
            cur = self.stack[-1]
            # 判断栈顶元素是否为数字
            if cur.isInteger():
                return True
            # 将列表弹出-展开-逆序入栈
            self.stack.pop()
            for i in range(len(cur.getList())-1, -1, -1):
                self.stack.append(cur.getList()[i])
        return False
    # 递归：利用队列在初始化时候递归展开所有子列表
    """
    # 深度优先搜索
    def dfs(self, nests):
        for nest in nests:
            # 判断是否为数字，是数字入队列尾部，是列表继续递归
            if nest.isInteger():
                self.queue.append(nest.getInteger())
            else:
                self.dfs(nest.getList())
                    
    def __init__(self, nestedList):
        # 初始化过程中展开所有列表
        self.queue = collections.deque()
        self.dfs(nestedList)

    def next(self):
        # 队列方法：popleft()弹出最左侧元素
        return self.queue.popleft()

    def hasNext(self):
        # 因为已经全部展开所以就看当前队列内是否还有元素
        return len(self.queue)
    """