# K 连续位的最小翻转次数
# 这是困难题光是题解都看了好久
class Solution(object):
    def minKBitFlips(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        # 第一种方法：差分数组
        n = len(A)
        # 定义差分数组：diff[i]=A[i-1]-A[i]记录变化值
        diff_list = [0 for _ in range(n + 1)]
        res = 0
        # 翻转次数变化值
        reverse_cnt = 0
        for i in range(n):
            # 将翻转次数累加到差分数组中，上台阶
            reverse_cnt += diff_list[i]
            # 判断经过前面的翻转之后，当前是0
            if (A[i] + reverse_cnt) % 2 == 0:   
                #超界了，则完不成
                if i + K > n:                   
                    return -1
                # 翻转次数
                res += 1
                # 左侧位置+1，体现在reverse_cnt上传递给差分数组实现翻转
                reverse_cnt += 1
                # 下台阶
                diff_list[i + K] -= 1
        return res

        # 第二种方法：滑动窗口
        """
        N = len(A)
        # 使用队列模拟滑动窗口：该滑动窗口的含义是前面 K - 1个元素中，以哪些位置起始的子区间进行了翻转
        que = collections.deque()
        res = 0
        for i in range(N):
            # 保持队列长度大小为K
            if que and i >= que[0] + K:
                que.popleft()
            # 判断当前元素是否需要翻转，队列元素个数代表i被前面K-1个元素翻转的次数
            # 0翻转偶数次变0还需翻转，1翻转奇数次变0还需翻转
            if len(que) % 2 == A[i]:
                if i +  K > N: return -1
                # 需要翻转就存储到队列中
                que.append(i)
                res += 1
        return res
        """