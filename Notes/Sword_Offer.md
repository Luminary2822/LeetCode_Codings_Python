## 剑指Offer（第2版）

#### 1.字符串的排列（38-Medium）

题目：https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/

方法：回溯法

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        # 回溯法
        res = []
        def backtrace(s, path):
            # 这种去重会超时，用set去重就不会超时
            # if not s and path not in res:
            if not s:
                res.append(path)
            for i in range(len(s)):
                # 当前选择的s[i]元素和其他元素的全排列组合，继续回溯
                backtrace(s[:i] + s[i+1:], path + s[i])
        backtrace(s, "")
        return list(set(res))
```

#### 2.二进制中1的个数(15-Easy)

题目：https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/

方法：位运算：将最低位1变成0

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n = n & (n - 1)
            count += 1
        return count
```

#### 3.在排序数组中查找数字I（53-Easy）

题目：https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/

方法：二分查找

```python
class Solution:
    def search(self, nums, target):
        # 二分法，从左右两方向向target逼近
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right-left) // 2
            if nums[mid] > target:
                right = mid - 1
            if nums[mid] < target:
                left = mid + 1
        
            # 先用二分法找到mid为target时，left和right向target靠近，寻找左右边界
            if nums[mid] == target:
                if nums[left] == target and nums[right] == target:
                    return right - left + 1
                if nums[left] < target:
                    left += 1
                if nums[right] > target:
                    right -= 1  
        return 0
```

#### 4.连续子数组的最大和(42-Easy)

题目：https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/

方法：动态规划/前缀和

```python
class Solution:
    def maxSubArray(self, nums):
        # dp[i]表示以nums[i]结尾的最大子数组和
        N = len(nums)
        dp =[0 for _ in range(N)]
        dp[0] = nums[0]
        for i in range(1, N):
            dp[i] = max(dp[i-1] + nums[i], nums[i])
        return max(dp)
        # # 前缀和
        # res = 0
        # ans = float('-inf')
        # # 计算前缀和，实时更新最值，当前缀和小于0时重新计算
        # for i in range(len(nums)):
        #     res += nums[i]
        #     ans = max(res, ans)
        #     if res < 0:res = 0
        # return ans
```

#### 5.两个链表的第一个公共节点(52-Easy)

题目：https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/

方法：双指针

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # 双指针
        if headA is None or headB is None:
            return None
        left, right = headA, headB
        # 如果有一个链表遍历到结尾的时候转换到另一个链表的头部，消除长度差值，最后相遇在公共节点。
        while left != right:
            left = left.next if left else headB
            right = right.next if right else headA
        return left
```

#### 6.链表中倒数第k个节点(22-Easy)

题目：https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/

方法：快慢指针

```python
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        # 特判
        if not head or not head.next:
            return head
        # 快慢指针初始化：距离为k，slow指向第一个结点，fast指向第k+1个结点
        slow, fast = head, head
        for _ in range(k):
            fast = fast.next
        # 同时向前走：当fast指向空时，slow指向第k个结点
        while fast:
            fast = fast.next
            slow = slow.next
        return slow
```

#### 7.斐波那契数列(10-Easy)

题目：https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/

方法：动态规划

```python
class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = (dp[i-1] + dp[i-2]) % 1000000007
        return dp[n]
```

#### 8.左旋转字符串(58-Easy)

题目：https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/

方法：局部旋转+整体旋转

```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        # 局部翻转+整体翻转
        # 先翻转前n，再翻转后n，最后整体翻转

        # 列表方法：切片+reverse
        s = list(s)
        s[:n] = list(reversed(s[:n]))
        s[n:] = list(reversed(s[n:]))
        s.reverse()
        return "".join(s)

        # 无调用函数方法：求余取模运算
        res = ""
        for i in range(n, n + len(s)):
            res += s[i % len(s)]
        return res
```

#### 9.数组中重复的数字（03-Easy）

题目：https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/

方法：哈希表/原地排序

```python
# 时间优先：哈希表
class Solution:
    def findRepeatNumber(self, nums):
        hashTable = {}
        for num in nums:
            if num in hashTable:
                return num
            else:
                hashTable[num] = hashTable.get(num,0) + 1
        return -1
```

```python
# 空间优先：原地排序
class Solution:
    def findRepeatNumber(self, nums) :
        # 通过交换操作，使元素的索引与值一一对应，通过索引映射对应的值，起到字典的作用
        # while和for不同的是，while的话交换完之后当前i不是直接加1要再走一遍循环判断是否与前面有重复
        i = 0
        while i < len(nums):
            # 数字已在对应索引位置无需交换，直接跳过
            if nums[i] == i:
                i += 1
                continue
            # 第二次遇到nums[i]，索引处已有值记录说明是重复元素，返回即可
            if nums[nums[i]] == nums[i]:return nums[i]
            # 交换至索引处
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        return -1
```

#### 10.二维数组中的查找（04-Medium）

题目：https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/

方法：标志位法，选右上角，往下走增大，往左走减小，可选。

```python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        # 标志位法
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        m = len(matrix)
        n = len(matrix[0])
        # 从右上角开始遍历，往下走增大，往左走减小
        row = 0
        col = n - 1
        while row < m and col >= 0:
            if matrix[row][col] > target:
                col -= 1
            elif matrix[row][col] < target:
                row += 1
            elif matrix[row][col] == target:
                return True
        return False
```

#### 11.替换空格（05-Easy）

题目：https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/

方法：预先扩容，从后向前

```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        # 扩充数组利用双指针法从后向前替换空格

        # 扩充数组到每个空格替换成"%20"之后的大小
        # 每碰到一个空格就多拓展两个格子，1 + 2 = 3个位置存’%20‘
        space_num = s.count(' ')
        res = list(s)
        res.extend([' '] * space_num * 2)

        # 从原字符串的末尾和扩展后字符串的末尾，双指针向前遍历
        left = len(s) - 1
        right = len(res) - 1

        while left >= 0:
            # 左指针指向不是空格，则复制字符到右指针
            if res[left] != ' ':
                res[right] = res[left]
                right -= 1
            # 左指针指向的是空格，右指针向前三个格填充%20
            else:
                res[right - 2: right + 1] = '%20'
                right -= 3 
            left -= 1
        return "".join(res)
```



## 剑指Offer专项突击版

#### 1.每日温度(38-Medium)

题目：https://leetcode-cn.com/problems/iIQa4I/

方法：单调栈

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # 维护从栈底到栈顶的单调递增栈
        N = len(temperatures)
        stack = []
        res = [0] * N
        for i in range(N):
            # 当前气温高于栈顶气温，计算等待天数，计算完毕后弹出
            while stack and temperatures[i] > temperatures[stack[-1]]:
                res[stack[-1]] = i - stack[-1]
                stack.pop()
            # 当前气温低于栈顶气温，入栈
            stack.append(i)
        return res
```

#### 2.爬楼梯的最少成本(88-Easy)

题目：https://leetcode-cn.com/problems/GzCJIP/

方法：动态规划

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # dp[i]表示爬到第i层需要花费的体力值
        N = len(cost)
        dp = [0] * N
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2, len(cost)):
            dp[i] = min(dp[i-1],dp[i-2]) + cost[i]
        return min(dp[N-1],dp[N-2])
```

