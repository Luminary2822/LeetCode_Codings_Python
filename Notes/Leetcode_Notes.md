## 每天做的题都在丢在这里面

#### 1. 计算质数（204-Easy）

题意：统计所有小于非负整数 n 的质数的数量。

分析：质数的定义：在大于 1 的自然数中，除 1和它本身以外不再有其他因数的自然数。

（1）枚举： [ 2, sqtr(n) ]

（2）埃及筛：如果 x 是质数，那么大于 x 的 x 的倍数 2x,3x, 一定不是质数

```python
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return 0
        cnt = 0
        # 设置一个都为false的数组
        flag = [ False ] * n
        for i in range(2, n):
            # 累计如果是false就是质数
            if flag[i] == False:
                cnt += 1
                # 从i*i开始筛选i的倍数设置为合数，小于i的已经筛选过了
                for j in range(i*i, n, i):
                    flag[j] = True
        return cnt
# 解释说明：
# 筛5的倍数时，我们从5的2倍开始筛，但是5 * 2会先被2 * 5筛去， 5 * 3会先被3 * 5会筛去，5 * 4会先被2 * 10筛去，所以我们每一次只需要从i*i开始筛，因为(2，3,…,i - 1)倍已经被筛过了

```

#### **2.分割数组为连续子序列**（659-Medium）

##### Key：贪心

题意：一个按升序排序的整数数组 num（可能包含重复数字），将它们分割成一个或多个子序列，其中每个子序列都由连续整数组成且长度至少为 3 。  

分析：由于需要将数组分割成一个或多个由连续整数组成的子序列，因此只要知道子序列的最后一个数字和子序列的长度，就能确定子序列

**（1）哈希表+最小堆**

​	 x在数组中，如果存在以x-1为结尾长度为k的子序列，x可加入其中得到长度为k+1的子序列，如果不存在需要新建一个包含x的子序列，长度为1。x在数组中时，如果存在多个子序列以x-1结尾，应该将x加入其中最短的子序列。

​	哈希表的键为子序列最后一个数字，值为一个最小堆（满足堆顶元素是最小的），用来存储所有子序列的长度，即堆顶元素即为最小的子序列长度。

​	遍历数组到x时，判断哈希表是否存在以x-1为结尾的子序列，则取出以x-1结尾的最小的子序列长度，长度加1作为以x为结尾的子序列长度，此时以x-1的子序列减少了一个，以x为结尾的子序列增加了一个。如果哈希表不存在以x-1结尾的子序列，则新建一个长度为1的以x结尾的子序列。遍历结束知乎，检查哈希表中存储的每个子序列长度是否都不小于3，即可判断是否可以完成分割，只要遍历每个最小堆的堆顶元素，即可判断每个子序列的长度是否都不小于3   

```python
# 升序数组分割一个或者多个子序列，每个子序列将由征连续整数且长度至少为3
# 哈希表 + 最小堆
import collections
import heapq
class Solution(object):
    def isPossible(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 创建一个列表字典键为子序列结尾值，值为子序列的长度（用最小堆存储，堆顶为最小子序列长度）
        # defaultdict 创建默认值
        mp = collections.defaultdict(list)
        for x in nums:
            # 如果存在以x-1为结尾的子序列，queue为以x-1为结尾的子序列对应的最小堆，存储着不同的长度
            # python3.8可以用海象运算符：queue := mp.get(x-1)
            queue = mp.get(x - 1)
            if queue :
                # 获取到以x-1结尾的最小子序列长度
                prevLength = heapq.heappop(queue)
                # 以x结尾的子序列的最小堆，push进一个长度值+1
                heapq.heappush(mp[x], prevLength + 1)
            else:
              # 不存在以x-1为结尾的子序列就创造一个以x为结尾的子序列，对应的值最小堆为 1
                heapq.heappush(mp[x], 1)
        # 遍历每个最小堆（判断子序列的存在和最小长度与3的比较）：
        # 每个最小堆满足：存在，且堆顶元素[0]大于3返回False
        # 所有最小堆都满足为False，经过any为False，再经过not为True表示分割成功
        # 若有一个最小堆不满足为True，经过any为True，再经过not为False表示分割失败
        return not any(queue and queue[0] < 3 for queue in mp.values())
```

**（2）贪心** 

​	使用两个哈希表，第一个哈希表存储数组中的每个数字的剩余次数，第二个哈希表存储数组中每个数字作为结尾的子序列的数量。遍历数组初始化第一个哈希表，之后遍历数组x，只有当一个数字的剩余次数大于 0 时，才需要考虑这个数字是否属于某个子序列。所以先判断x在第一个数组中的剩余次数是否大于0，判断是否存在以x-1为结尾的子序列，即判断第二个哈希表数量是否大于0.

​	如果大于0将x加入子序列中，第一个表x剩余次数-1，第二个表以x-1为结尾的子序列数量-1，以x为结尾的子序列数量+1；

​	如果小于0不存在的话x即为一个子序列第一个数，为了满足得到长度至少为3的子序列，x+1和x+2必须在子序列中，因此要判断第一个哈希表x+1和x+2的剩余次数是否大于0，大于0的话可以新建一个长度为3的子序列，由于三个数被使用所以需要在第一个哈希表将三个数的剩余次数分别减1，最后以x+2结尾，所以第二个哈希表将x+2作为结尾的子序列数量+1，如果数组遍历结束时，没有遇到无法完成分割的情况，可以完成分割返回True，否则返回False

```python
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
```

#### **3.合并二叉树**（617-Easy）

题意：给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

分析：深度优先搜索，可以两种方法：改变或不改变原来二叉树结构，记得判断根节点是否存在

```python
class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        # 改变二叉树结构，合并到t1树上
        if t1 and t2:
            t1.val += t2.val
            t1.left = self.mergeTrees(t1.left, t2.left)
            t1.right = self.mergeTrees(t1.right, t2.right)
         # 如果t1和t2有一个不存在则返回另外一个
        return t1 or t2
```

#### 4.杨辉三角（118-Easy)

题意：给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。

分析：首先每行生 成为1的列表，然后设置一个指向前一行，遍历当前行从1-i（不包括0和i），对应位置的值由上一行的前一个位置和当前位置值相加可得。将当前行不断添加到结果列表。

```python
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        result = []
        # 从0开始遍历到numRows
        for i in range(numRows):
            # 每行生成对应个数全为1的列表
            now = [1] * (i+1)
            for n in range(1, i):
                # 当前位置等于前一行当前位置-1元素值+前一行当前位置值
                now[n] = pre[n-1] + pre[n]
            # 以列表嵌套的形式加入结果列表中
            result += [now]
            # 前一行的指向移动
            pre = now
        return result
```

#### 5.汉明距离（461-Easy）

题意：两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。给出两个整数 x 和 y，计算它们之间的汉明距离。

分析：两个数先做异或操作，相同为0不同为1，然后计算结果中1的个数

```python
class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        # 异或：相同为0,不同为1，看结果中1的个数
        n = x ^ y
        cnt = 0
        # n不为0的情况下可以计数出有多少1
        while n:
            # 按位与的作用：将n的二进制表示中最低位的1改成0
            n &= (n-1)
            cnt += 1
        return cnt

```

#### 6.子集（78-Medium）

##### Key：回溯法-求不含重复元素的数组的子集

题意：给定一组不含重复元素的整数数组 *nums*，返回该数组所有可能的子集（幂集）。说明：解集不能包含重复的子集。

分析：实际上应该属于一道回溯类型的题，但是最开始看到比较容易理解的思路是从前往后遍历, 每遇到一个数, 之前的所有集合添加上这个数，组成新的子集。学了回溯之后再附上回溯的方法

```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # 从前往后遍历, 遇到一个数, 之前的所有集合添加上这个数, 组成新的子集.
        res = [[]]
        # 从前往后遍历所有数
        for i in range(len(nums)):
            # 遍历之前的所有集合
            size = len(res)
            for j in range(0,size):
                temp = list(res[j])
                # 之前的每个集合都加上新的数
                temp.append(nums[i])
                # 组成新的集合再加入结果集
                res.append(temp)
        return res
    	# 现在会回溯了，补充一款回溯写法：
        """
        def subsets(self, nums):
            res = []
            self.dfs(nums, 0, res, [])
            return res
        def dfs(self, nums, index, res, path):
            res.append(path)
            # 在nums后续数字中依次选择加入到路径当中
            for i in range(index, len(nums)):
                self.dfs(nums, i + 1, res, path + [nums[i]])
        """
```

#### 7.存在重复元素（217-Easy）

题意：给定一个整数数组，判断是否存在重复元素。如果任意一值在数组中出现至少两次，函数返回 `true` 。如果数组中每个元素都不相同，则返回 `false` 。

分析：直接用python的set函数，将给定的数组存入set判断与原数组的长度是否匹配。实际上官方的题解是有两个方法：

（1）排序：对数字从小到大排序，数组的重复元素一定出现在相邻位置中，因此可以扫描已排序的数组，每次判断相邻的两个元素是否相等，如果相等则说明存在重复的元素。

（2）哈希表：对于数组中每个元素，我们将它插入到哈希表中。如果插入一个元素时发现该元素已经存在于哈希表中，则说明存在重复的元素。

```python
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # set() 函数创建一个无序不重复元素集
        new_nums = list(set(nums))
        # 判断set和原数组长度是否相同，不同说明有重复元素返回true，相同说明没有重复元素返回false
        return len(new_nums) != len(nums)
        # 这里直接一行解决问题
        # return len(set(nums)) != len(nums)
```

#### 8.找不同（389-Easy）

题意：给定两个字符串 s 和 t，它们只包含小写字母。字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。请找出在 t 中被添加的字母。

分析：Counter方法和位运算比较经典

（1）可用`collections`里的`Counter`类构造字符和出现次数的哈希表，利用相减得到差异字符哈希表再调用`elements`方法获取到差异字符，可用next迭代输出或转化为List按列表输出

（2）位运算：0异或任何非0数都为它本身，将两个字符串拼接成一个字符串，则问题转换成求字符串中出现奇数次的字符，遍历逐个异或s和t，最后的值即为额外的元素。

（3）利用ASCII值，每个字符串的ASCII求和，两个和相减的差值即代表被添加的字符。

```python
import collections
class Solution(object):
# 第一种：利用Counter相减方法
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # 建立字符与个数的键值对
        num1 = collections.Counter(t)
        num2 = collections.Counter(s)
        # 获得差异字符的键值对
        diffCounter = num1 - num2
        # elements方法将键列出转成list获取差异字符
        diff = list(diffCounter.elements())[0]
        return diff

#第二种：Counter+next迭代的方法
    """
    return next((Counter(t)-Counter(s)).elements())
    """

#第三种：遍历 Counter里面的key
    """
    def findTheDifference(self, s: str, t: str) -> str:
        dict_s = collections.Counter(list(s))
        dict_t = collections.Counter(list(t))
    for key in dict_t.keys():
        # 这里的or是为了ae和aea这样的情况，t内新增加的差异与已有字符重复，key也在s内，但是值不同。
        if key not in dict_s.keys() or dict_t[key] != dict_s[key]:
            return key
    """

#第四种：位运算
"""
class Solution(object):
    def findTheDifference(self, s, t):
        res = 0    # 0^任何非0数都为它本身
        for a in s:
            res ^= ord(a)   # ord() 是将一个字符转化为ASCII值 
        for a in t:
            res ^= ord(a)
        return chr(res)     #将额外元素的ASCII转化为字符
"""



```

#### 9.翻转二叉树（226-Easy）

题意：翻转一棵二叉树

分析：先交换左右子树，递归

```python
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root == None:
            return root
        # 先交换左右子树
        root.left, root.right = root.right, root.left
        # 递归交换左右子树
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

#### 10.旋转图像（48-Medium）

题意：给定一个 n × n 的二维矩阵表示一个图像。将图像顺时针旋转 90 度。你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

分析：可以直接利用旋转库函数`*np.rot90(matrix,-1)*`，或者利用转置和水平镜像的库函数。也可以找到坐标之间的关系，开始的对应列旋转过后是对应行，开始对应的行旋转过后是对应倒数第几列：`matrix_new[j][n-i-1] = matrix[i][j]`

```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        # 本题不用return，直接修改matrix
        n = len(matrix)
        matrix_new = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix_new[j][n-i-1] = matrix[i][j]
        matrix[:] = matrix_new

        # 第二种方法：充分利用库函数的便捷方法：直接修改matrix，利用numpy顺时针-1旋转90度函数
        """
        import numpy as np
        matrix[:]=np.rot90(matrix,-1).tolist()
        """

        # 第三种方法：先矩阵转置再水平镜像翻转即可
        """
        # 矩阵转置
        matrix_trans = np.transpose(matrix)
        # 矩阵镜像fliplr水平方向，flipud垂直方向
        res = np.fliplr(matrix_trans)
        """
```

#### 10.去除重复字母（316-Medium）

题意：给你一个字符串s，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证返回结果的字典序最小（要求不能打乱其他字符的相对位置）。示例输入：`"cdadabcc"`；输出：`"adbc"`

分析：保证字典序最小，保证相对位置。字典序：是指按照单词出现在字典的顺序比较两个字符串的方法，例如`abc`的字典序在`acbd`的前面。建立字符与其出现次数的键值对哈希表，建立一个list作为栈，将字符依次入栈，如果是新字符栈内没有的，先遍历和栈内的元素比较字典序，如果栈内元素字典序较大且剩余次数>0即弹出pop它两个条件必须同时满足的时候才pop出去，如果剩余次数为0不能pop字典序小也不能pop，接着新字符入栈，对应次数减一。

```python
class Solution(object):
    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        # 一个用列表实现的栈,pop和append都是列表的方法
        stack = []
        c = collections.Counter(s)
        for i in s:
            if i not in stack:
                while stack and stack[-1] > i and c[stack[-1]] > 0:
                    stack.pop()
                stack.append(i)
            c[i] -= 1
        return ''.join(stack)
```

#### 11.二叉树的锯齿形层序遍历（103-Medium）

##### Key：BFS广度优先遍历

题意：给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。输入[3,9,20,null,null,15,7]，输出[[3],[20,9],[15,7]]。

分析：BFS广度优先遍历的变种题目，层数需要做标记，先从左到右，下一层从右到左交替进行。可以采用异或来标记层，flag为1的顺序输出，为0的逆序输出

```python
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root == None:
            return []
        # flag为0的层需要倒序，利用异或方法来分别
        flag = 1
        # 用来存储每一层结点的列表
        bfs = [root]
        # 结果集
        res = []
        # 当bfs非空时，获取到当前层的结点值，并且将下一层记录下来
        while bfs:
            # 用来存储下一层结点的临时列表
            temp = []
            # 存储结果集的每个小列表表示每一层的结点值
            vals = []
            for n in bfs:
                vals.append(n.val)
                if n.left:
                    temp.append(n.left)
                if n.right:
                    temp.append(n.right)
            # 根据flag判断当前层的结点值是否需要逆序
            vals = vals if flag else vals[::-1]
            # 将当前层加入结果集
            res.append(vals)
            # 更新到下一层结点继续遍历
            bfs = temp
            # 利用异或设置flag的值
            flag ^= 1
        return res
```

#### 12.分发糖果（139-Hard）

题意：老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。需要按照以下要求，帮助老师给这些孩子分发糖果：每个孩子至少分配到 1 个糖果。相邻的孩子中，评分高的孩子必须获得更多的糖果。那么这样下来，老师至少需要准备多少颗糖果呢？

分析：两次遍历，从前到后和从后到前，从前到后遍历每个小朋友先和前一个小朋友比较分数调整糖果，从后到前遍历每个小朋友和后一个小朋友比较分数调整糖果。

```python
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

```

#### 13.同构字符串（205-Easy）

题意：给定两个字符串 s 和 t，判断它们是否是同构的。如果 s 中的字符可以被替换得到 t，那么这两个字符串是同构的。所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。

分析：哈希表简建立映射关系，位置索引

（1）定义哈希表（字典）建立两个字符串的映射关系，如果对应字符的映射关系不成立的话就返回`False`，键存在于字典中但是对应的值不匹配（s中有重复字母t中不存在）以及键不存在于字典中但是对应值在字典中（s中没有重复字母t中存在）。

（2）位置索引：同构字符串每个字符出现的首次位置应相同，若不同则`False`

```python
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # 第一种：哈希表解法
        hashMap = {}
        # zip方法打包成元组，建立两个字符串的键值对关系
        for i,j in zip(s, t):
            # s = "foo", t = "bar"，o键已经映射到r，不能再映射到a
            if i in hashMap and hashMap[i] != j:
                return False
            # s = "list", t = "fool"，s键不存在的情况下，o值已经被i映射，存在于字典中，不能再建立映射
            elif i not in hashMap and j in hashMap.values():
                return False
            # 正常情况就加入字典中
            hashMap[i] = j
        return True

    	# 第二种：索引解法：同构字符串，每字符首次出现、最后出现、指定位出现索引始终相同
        """
        # 从字符串中找出某个子字符串第一个匹配项的索引位置
        return [s.index(i) for i in s] == [t.index(i) for i in t]
        """

```

#### 14.最后一块石头的重量（1046-Easy）

##### Key：利用数据相反数构建最大堆

题意：有一堆石头，每块石头的重量都是正整数。每一回合，从中选出两块最重的石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：如果 x == y，那么两块石头都会被完全粉碎；如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。

分析：首先想到的是很简单的方法，当列表剩余元素超过一个时，列表逆序排序，取出前两个为最重的，相减的值再添加到列表当中（0也没有关系）。官方题解：用堆函数的思想，堆是可以自动排序的数据结构，python只有最小堆，所以要取数据的相反数将列表构成堆，然后pop出负数最小的实际上是最大的数，判断是否相等把相减的值push进堆里面，最后输出堆顶。

```python
class Solution(object):
    def lastStoneWeight(self, stones):
        """
        :type stones: List[int]
        :rtype: int
        """
        第一种：简单解法：先降序，取前两个大的相减的值再加入列表，最后剩余一个输出
        while len(stones) > 1:
            # 降序排序
            stones.sort(reverse = True)
            stones.append(stones[0] - stones[1])
            del stones[0:2]
        return stones[0]
		
        # 第二种：最小堆（可以自动排序）转换，python只支持小顶堆，所以在入堆的时候要添加的是数据的相反数，这样堆顶就会对应最大的元素
        """
        #形成一个数据相反数的列表
        heap = [-stone for stone in stones]
        #让列表具备堆特征
        heapq.heapify(heap)

        # 模拟堆的操作
        while len(heap) > 1:
            x,y = heapq.heappop(heap),heapq.heappop(heap)
            if x != y:
                heapq.heappush(heap,x-y)
        if heap: return -heap[0]
        return 0
        """
```

#### 15.滑动窗口最大值（259-Hard)

##### Key 1：滑动窗口问题：对于长度为n窗口为k，窗口数量为n-k+1

##### Key 2：去除最大值在滑动窗口左侧思想，相邻窗口共用k-1个元素

题意：给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。返回滑动窗口中的最大值。

分析：如果直接按照题意写会超时，所以看了官方题解用优先队列的思想，采用构建最大堆自动排序来确定最大值，需要判断当前队列的最大值是否在当前的滑动窗口内，判断方法：当前最大值若不在滑动数组汇总，那在nums中的位置定在滑动窗口左边界的左侧，如果不在后续向右滑动也用不到这个值可以永久从优先队列中移除，直到堆顶元素出现在滑动窗口中，即可加入结果集作为当前窗口的最大值。关键：要在队列中存储二元组（-nums[i]，i）表示元素以及它在数组中的下标。

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # 官方题解：利用堆构建优先队列
        n = len(nums)
        # 注意 Python 默认的优先队列是小根堆，构建k个数值与坐标构成元组的列表
        q = [(-nums[i], i) for i in range(k)]
        # 将列表具有堆的性质
        heapq.heapify(q)
        # 先加入-nums[0]，初始最大值
        res = [-q[0][0]]
        for i in range(k, n):
            # 队列加入一个新元素
            heapq.heappush(q, (-nums[i], i))
            # 判断当前堆顶元素（即为最大值）是否在滑动窗口内，如果不在的话它的坐标一定在滑动窗口左边界的左侧
            while q[0][1] <= i - k:
                # 后续移动滑动窗口时这个值一定不会出现了所以可以永久从优先队列删除
                heapq.heappop(q)
            # 结果集加入当前堆顶元素数值
            res.append(-q[0][0])
        return res

        # 第二种：直接理解题意法（1）(2)-会超时
        """
        result = []
        if len(nums) == 0:
            return
        # 遍历每个滑动窗口寻找最大值
        for i in range(0, len(nums)-k+1):
            result.append(max(nums[i:k+i]))
        return result
        # （2）
        if not len(nums) and not k: return []
        rst = []
        # 先构建初始窗口，选取最大值
        tmp = nums[0:k]
        rst.append(max(tmp))
        # 遍历从k开始到后面的数值，每次窗口第一个元素删除，添加一个新元素，寻找最大值加入结果列表
        for i in nums[k:]:
            del tmp[0]
            tmp.append(i)
            rst.append(max(tmp))
        return rst
        """
```

#### 17.分隔链表（86-Medium）

题意：给你一个链表和一个特定值 x ，请你对链表进行分隔，使得所有小于 x 的节点都出现在大于或等于 x 的节点之前。你应当保留两个分区中每个节点的初始相对位置。

分析：新建两个链表，分别存储比x小的和大于等于x的结点，最后将两个链表连接起来，注意新建结点值为0，.next才是有效结点。

```python
class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        # 两个链表分别存储比x小,与x相等或者比x大的元素，再将两个链表连接起来
        # 设置两个链表的值为0的头结点，less和more是移动的指针
        less_head = less = ListNode(0)
        more_head = more = ListNode(0)
        while head:
            if head.val < x:
                less.next = head
                less = less.next
            else:
                more.next = head
                more = more.next
            head = head.next
        # 设置最后结点指向为空
        more.next = None
        # 链表连接：more_head为自己初始化的结点值为0，要指向next才是真的值
        less.next = more_head.next
        # 返回连接后链表的结点，less_head为自己初始化的结点值为0
        return less_head.next
```

#### 18.数组中的第K个最大元素（215-Medium）

##### Key：切分函数partition + 优先队列

题意：在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

分析：有三种解法，一是调用库函数，先排序再找目标的索引为len-k。二是partition操作减而治之，经过一次partition操作，总能排定一个元素还能知道元素所在的位置，左边都不大于它右边都不小于它。三为优先队列维护k个元素的最小堆，堆不满的时候新读到的数大于堆顶的情况下添加，堆自己内部调整结构，最后输出堆顶即为第k大元素；优先队列若维护len个元素的话，有以下两种思路可取：

思路1：把 len 个元素都放入一个最小堆中，然后再 pop() 出 len - k 个元素，此时最小堆只剩下 k 个元素，堆顶元素就是数组中的第 k 个最大元素。

思路2：把 len 个元素都放入一个最大堆中，然后再 pop() 出 k - 1 个元素，因为前 k - 1 大的元素都被弹出了，此时最大堆的堆顶元素就是数组中的第 k 个最大元素。

```python
import heapq
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # 第一种：利用堆构造k个容量的优先队列
        size = len(nums)
        ambdaL = []
        for i in range(k):
            heapq.heappush(L, nums[i])
        for j in range(k,size):
            if nums[j] > L[0]:
                # 弹出最小的元素，并将x压入堆中
                heapq.heapreplace(L, nums[j])
        return L[0]

        # 第二种：简单调用库函数排序获取第K大的值
        """
        size = len(nums)
        nums.sort()
        return nums[size-k]
        """
        
        # 第三种：切分操作：减而治之
        """
        # 设置左右指针和目标位置
        size = len(nums)
        left = 0
        right = size - 1
        target = size - k

        while True:
            # 进行一次切分操作，得到确定某元素的位置比较是否与目标一致
            index = self._partition(nums, left, right)
            if index == target:
                return nums[index]
            # 不一致则判断位置在目标位置左侧还是右侧以此调节指针
            elif index < target:
                left = index + 1
            else:
                right = index - 1
    # 切分函数
    def _partition(self, nums, left, right):
        pivot = nums[left]
        j = left
        for i in range(left + 1, right + 1):
            if nums[i] < pivot:
                j += 1
                nums[i], nums[j] = nums[j], nums[i]
        # 在之前遍历的过程中，满足 [left + 1, j] < pivot，并且 (j, i] >= pivot
        nums[left], nums[j] = nums[j], nums[left]
        # 交换以后 [left, j - 1] < pivot, nums[j] = pivot, [j + 1, right] >= pivot
        return j
        """
```

#### 19.缀点成线（1232-Easy）

题意：在一个 XY 坐标系中有一些点，我们用数组 coordinates 来分别记录它们的坐标，其中 coordinates[i] = [x, y] 表示横坐标为 x、纵坐标为 y 的点。请你来判断，这些点是否在该坐标系中属于同一条直线上，是则返回 true，否则请返回 false。

分析：数学问题，先拿出两个点，再遍历后面第三个点依次与两个点计算斜率，避免斜率无穷大和斜率不存在的情况，将求斜率公式改为交叉相乘来判断。这题我用set一直有测试用例通不过，一个简单题刷刷刷的在降低通过率。

```python
class Solution(object):
    def checkStraightLine(self, coordinates):
        """
        :type coordinates: List[List[int]]
        :rtype: bool
        """
        if len(coordinates) <= 2:
            return True
        x0, y0 = coordinates[0]
        x1, y1 = coordinates[1]
        for x, y in coordinates[2:]:
            if (y0 - y1) * (x1 - x) != (x0 - x1) * (y1 - y):
                return False
        return True
```

#### 20.公平的糖果棒交换（888-Easy）

题意：爱丽丝和鲍勃有不同大小的糖果棒：A[i] 是爱丽丝拥有的第 i 根糖果棒的大小，B[j] 是鲍勃拥有的第 j 根糖果棒的大小。因为他们是朋友，所以他们想交换一根糖果棒，这样交换后，他们都有相同的糖果总量。（一个人拥有的糖果总量是他们拥有的糖果棒大小的总和。）返回一个整数数组 ans，其中 ans[0] 是爱丽丝必须交换的糖果棒的大小，ans[1] 是 Bob 必须交换的糖果棒的大小。

分析：两个数组求和相加除2就是最后分配完成之后两个数组最后的糖果棒数，计算A数组距离最终数组还差多少，遍历A数组每个元素+差值看该数值是否在B中，在的话则交换数组中即为遍历A的当前元素和B中该元素。

```python
class Solution(object):
    def fairCandySwap(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        final_length = (sum(A) + sum(B)) / 2
        num_A = final_length - sum(A)
        for a in A:
            if num_A + a in B:
                return [a, int(a+num_A)]
```

#### 21.有效的括号（20-Easy）

题意：给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。有效字符串需满足：左括号必须用相同类型的右括号闭合。左括号必须以正确的顺序闭合。

分析：建立一个括号匹配的字典，利用栈的思想进行匹配遇到左括号入栈，遇到右括号将当前栈顶在括号字典中对应的与当前右括号匹配，若匹配失败则无效，最后栈空为有效

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 如果字符串长度为奇数，那一定不是
        if len(s) %2 == 1:
            return False
        # 建立一个括号映射字典
        pire = {'(':')', '{':'}', '[':']'}
        stack = []
        # 遍历字符串
        for char in s:
            # 判断是否为左括号，左括号入栈
            if char in pire:
                stack.append(char)
            # 非左括号即为右括号，弹出栈应该匹配的有括号看是否与当前字符匹配
            else:
                if not stack or pire[stack.pop()] != char:
                    return False
        # 最后若栈为空返回True，非空返回False
        return True if not stack else False

        # 一种比较简单易懂的方法，判断这些成对的括号是否出现在s中，若出现则全部替换为空，最后判断s是否为空
        """
        while '{}' in s or '()' in s or '[]' in s:
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''
        """
```

#### 22.替换后的最长重复字符（424-Medium）

##### Key：滑动窗口

题意：给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 k 次。在执行上述操作后，找到包含重复字母的最长子串的长度。

分析：滑动窗口问题，设置左右指针，右指针移动记录字符出现的次数，判断窗口长度-出现过最多次的字符数量与k的比较，如果比k大就移动左指针，实时记录窗口的最大长度。

```python
class Solution(object):
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        # 设置左指针和结果长度
        left, res= 0, 0
        # 设置字典存储每个字符的出现次数
        cur = collections.defaultdict(int)
        # 遍历字符串中每个字符和下标位置
        for right, val in enumerate(s):
            # 存储出现次数
            cur[val] += 1
            # 窗口内可替换字母数量大于K的时候，移动左指针向右
            while right - left + 1 - max(cur.values()) > k:
                cur[s[left]] -= 1
                left += 1
            # 获得当前窗口最大值
            res = max(res, right - left + 1)
        return res

```

#### 23.滑动窗口中位数（480-Difficult）

##### Key: 滑动窗口、二分查找

题意：中位数是有序序列最中间的那个数。如果序列的大小是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。[2,3,4]，中位数是 3，[2,3]，中位数是 (2 + 3) / 2 = 2.5。给你一个数组 nums，有一个大小为 k 的窗口从最左端滑动到最右端。窗口中有 k 个数，每次窗口向右移动 1 位。你的任务是找出每次窗口移动后得到的新窗口中元素的中位数，并输出由它们组成的数组。

分析：直接理解题意关键点在于要将原数组利用i去遍历，k个作为一组切片进行深拷贝，要求改变拷贝后的数组不影响原数组，然后排序计算中位数加入结果数组，这样效率较低。数组+二分查找的方法效率较高，有求中位数专门的公式，维护切片数组a不断移除和加入新的数，利用排序模块`bisect`插入到正确的位置上并利用公式计算中位数。求a的中位数公式：`median = lambda a: (a[(len(a)-1)//2] + a[len(a)//2]) / 2`

```python
class Solution(object):
    def medianSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[float]
        """
        # 第一种：直接阅读理解题意简单法，运行环境为python3
        res = []
        i = 0
        # k为奇数flag = true
        flag = True if k % 2 == 1 else False
        # 滑动窗口为i,i + k - 1
        while i + k - 1 < len(nums):
            # 在python3的环境下切片和copy都可以复制列表，tmp指向不同的对象
            # 切片和copy两种方法都可以实现得到两个指向不同对象独立的列表
            # tmp = nums[i:i+k].copy()
            tmp = nums[i:i+k]
            tmp.sort()
            # k为奇数中位数取中间值，为偶数取中间两个值的平均数
            if flag:
                res.append(tmp[k // 2])
            else:
                val = tmp[k // 2] + tmp[k // 2 - 1]
                res.append(val/2)
            i += 1
        return res

        # 第二种：数组+二分查找
        """
        # python排序模块
        import bisect
        # 求中位数方法，匿名函数lambda接受参数并返回表达式的值
        median = lambda a: (a[(len(a)-1)//2] + a[len(a)//2]) / 2
        # 维护数组a保存当前数组且有序
        a = sorted(nums[:k])
        res = [median(a)]
        # i表示删除值，j表示插入值，滑动窗口a中i删除，j插入在合适的位置，求中位数加入结果集
        # 注意：i的取值范围nums[:-k]去除后k个的元素的列表
        for i, j in zip(nums[:-k], nums[k:]):
            # bisect_left将数插入到正确的位置且不会影响之前的排序，left如果有重复数值插入在左边
            a.pop(bisect.bisect_left(a, i))
            a.insert(bisect.bisect_left(a, j), j)
            res.append(median(a))
        return res
        """
```

#### 24.子数组最大平均数I（643-Easy）

题意：给定 `n` 个整数，找出平均数最大且长度为 `k` 的连续子数组，并输出该最大平均数。

分析：求滑动窗口内均值比较大小，注意不要超时，每次向滑动窗口增加或者减少一个元素，计算平均值可以先计算求和，最后返回的时候再计算平均。

```python
class Solution(object):
	def findMaxAverage(self, nums: List[int], k: int) -> float:
        # 在python3的环境下运行结果正确
        # 先求和，再遍历从k到最后，对滑动窗口逐个添加和删除元素再选取最大和，最后求平均
        maxSum = sum(nums[0:k])
        res = maxSum
        for i in range(k, len(nums)):
            maxSum += nums[i] - nums[i - k]
            res = max(res,maxSum)
        return res / k
```

#### 25.最长湍流子数组（978-Meidum）

##### Key：双指针、动态规划

题意：当 A 的子数组 A[i], A[i+1], ..., A[j] 满足下列条件时，我们称其为湍流子数组：若 i <= k < j，当 k 为奇数时， A[k] > A[k+1]，且当 k 为偶数时，A[k] < A[k+1]；或 若 i <= k < j，当 k 为偶数时，A[k] > A[k+1] ，且当 k 为奇数时， A[k] < A[k+1]。也就是说，如果比较符号在子数组中的每个相邻元素对之间翻转，则该子数组是湍流子数组。返回 A 的最大湍流子数组的长度。

分析：湍流子数组的定义是元素的值的变化「减少」和「增加」交替出现，且相邻元素的值不能相等。可用双指针的动态规划两种方法求解。

（1）**双指针法**：数字表明前后方向，利用乘积判断是否为湍流数组，方向为0包含了数组长度为1的情况，不是湍流数组两种情况有相邻重复数值以及方向相同来判断左指针的移动，右指针始终遍历稳步向前，指针位置确定好之后更新一下前置方向，便于后续是否湍急判断，实时获取最大长度。

（2）**动态规划**：关键信息子数组是输入数组里 **连续** 的一部分，根据「减少」和「增加」交替出现的特点，规模较小的湍流子数组组成了规模较大的湍流子数组的长度，因此可以使用「动态规划」求解。定义状态以及状态转移方程，两个状态数组，分别表示以 i 结尾的在增长和降低的最长湍流子数组长度，初始化为 1。

​	状态：状态 dp[i] 为：以 i 位置结尾的最长连续子数组的长度；<br>	a.定义 up[i] 表示以位置 i 结尾的，并且 arr[i - 1] < arr[i] 的最长湍流子数组长度。<br>	b.定义 down[i] 表示以位置 i 结尾的，并且 arr[i - 1] > arr[i] 的最长湍流子数组长度

​	状态转移方程定义：<br>	a.up[i] = down[i - 1] + 1，当 arr[i - 1] < arr[i]<br>	b.down[i] = up[i - 1] + 1，当 arr[i - 1] > arr[i]

​	**初始化**：只有一个元素的时候，湍流子数组的长度是 1；
​	**输出**：两个状态数组所有元素的最大值是最长湍流子数组的长度；
​	**空间优化**：空间优化：当前阶段状态只和上一个阶段状态有关，可以对状态数组进行重复利用。

```python
class Solution(object):
    def maxTurbulenceSize(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        # 第一种双指针法：用数字表明方向，利用前后两方向乘积判断当前是否为湍流数组
        # 进而判断平坡还是方向相同来移动左指针，右指针稳步向前，实时更新前置方向和最大长度
        n = len(arr)
        left = 0
        # 数组长度为 1是无方向
        direction = 0
        maxLen = 1
        for right in range(1, n):
            # 如果不是湍流数组，那么前后两个方向乘积定>=0
            # 当前后方向相同或者平坡时（存在相邻的相同元素）
            if (arr[right] - arr[right - 1]) * direction >= 0:
                # 平坡，移动left指针越过重复元素到right
                if arr[right] == arr[right - 1]:
                    left = right
                # 方向相同，移动left到新方向的起点处
                else:
                    left = right - 1
            # 更新前置方向
            direction = arr[right] - arr[right-1]
            # 获取当前最大长度
            maxLen = max(maxLen, right - left + 1)
        return maxLen
        # 第二种：动态规划,利用连续规模较小子数组组成规模较大子数组。
        """
        N = len
        # 定义两个状态数组，初始化为 1
        up = [1] * N
        down = [1] * N
        res = 1
        for i in range(1, N):
            # 状态转移方程
            if arr[i-1] < arr[i]:
                up[i] = down[i-1] + 1
            elif arr[i - 1] > arr[i]:
                down[i] = up[i - 1] + 1
            # 获取最大长度
            res = max(res, max(up[i], down[i]))
        return res
        """
```

#### 26.最大子序和(53-Easy)

##### Key：动态规划

题意：给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

分析：动态规划简单题，最长连续子数组问题。找到状态定义以及状态转移方程。状态定义：dp[i]代表以nums[i]结尾的最大子序列和。状态转移方程：`dp[i] = max(dp[i-1] + nums[i], nums[i])`，更简便写法直接用nums[i]来存储以其结尾的最大子序列和，到上一项为止的最大子序列和num[i-1]与0比较决定是继续加还是取nums[i]本身。

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 动态规划
        # 状态定义：dp[i]代表以nums[i]结尾的最大子序列和
        # 转移方程：dp[i]等于dp[i-1]+nums[i]、nums[i]二者最大值
        dp = [nums[0]]
        for i in range(1, len(nums)):
            dp.append(max((dp[i-1]+nums[i]), nums[i]))
        return max(dp)
        # 简便写法,nums[i]更新为存储到目前为止最大子序列和
        # nums[i-1]为前一项最大子序列和，与0比较
        # 大于0则加上当前项赋予到nums[i]表示目前的最大子序列和，小于0则nums[i] = 自身
        """
        for i in range(1, len(nums)):
            # 等式左边nums[i]表示以其为结尾的最大子序列和
            # 右边第一个nums[i]为nums数组本身原值，nums[i-1]和第一个nums[i]内涵相同
            nums[i]= nums[i] + max(nums[i-1], 0)
        return max(nums)
        """
```

#### 27.K个不同整数的子数组(992-Hard)

##### Key：滑动窗口，恰好转最多，子数组长度与子数组个数

题意：给定一个正整数数组 A，如果 A 的某个子数组中不同整数的个数恰好为 K，则称 A 的这个连续、不一定独立的子数组为好子数组。（例如，[1,2,3,1,2] 中有 3 个不同的整数：1，2，以及 3。）返回 A 中好子数组的数目。

分析：三个关键点，问题转换（经验）- 利用滑动窗口结题 - 累加到res为子数组长度（意义为满足条件子数组个数）

（1）恰好由 K 个不同整数的子数组的个数 = 最多由 K 个不同整数的子数组的个数 - 最多由 K - 1 个不同整数的子数组的个数。

（2）滑动窗口思想：利用counter记录不同数字出现次数，变换左指针得到满足条件的子数组利用长度记录个数，右指针向前。

（3）`res += right - left + 1`:我们要求由最多 K 个不同整数组成的子数组的个数，那么对于长度 [left, right] 区间内的每个子数组都是满足要求的，res 要累加的其实是符合条件的并且以 right 为右端点的子数组个数，这个数量就是right - left + 1，即区间长度。

```python
class Solution(object):
    def subarraysWithKDistinct(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        # 结果 = 最多由 K 个不同整数的子数组的个数 - 最多由 K - 1 个不同整数的子数组的个数。
        return self.atMostK(A, K) - self.atMostK(A, K-1)
    # 求 A 中由最多 K 个不同整数组成的子数组的个数
    def atMostK(self, A, K):
        N = len(A)
        # 记录不同数字出现的个数
        counter = collections.Counter()
        # 左右指针
        left, right = 0, 0
        # 记录不同数字的个数
        distinct = 0
        res = 0
        while right < N:
            if counter[A[right]] == 0:
                distinct += 1
            counter[A[right]] += 1
            # 当不同数字个数大于K时，对应counter内个数减一，如果该数字个数为0则distinct减一，最后移动左指针向前
            while distinct > K:
                counter[A[left]] -= 1
                if counter[A[left]] == 0:
                    distinct -= 1
                left += 1
            # 结果集加入每个满足条件以right为右区间的子数组个数（即为长度）
            res += right - left + 1
            # right指针向前
            right += 1
        return res
    # 第二种atMostK函数：不用distinct利用k来控制个数
    """
    def atMostK(self, A, K):
        count = collections.Counter()
        res = i = 0
        for j in range(len(A)):
            if count[A[j]] == 0:
                K -= 1
            count[A[j]] += 1
            while K < 0:
                count[A[i]] -= 1
                if count[A[i]] == 0:
                    K += 1
                i += 1
            res += j - i + 1
    """
```

#### 28.字符串的排列(567-Medium)

##### Key：滑动窗口 + 排列思想问题转化 + 字典

题意：给定两个字符串 **s1** 和 **s2**，写一个函数来判断 **s2** 是否包含 **s1** 的排列。换句话说，第一个字符串的排列之一是第二个字符串的子串。

分析：如果字符串 `a` 是 `b` 的一个排列，那么当且仅当它们两者中的**每个字符的个数都**必须完全相等。等价问题转化 **s1** 的任意一种排列是 **s2** 的一个子串，不需要当成全排列问题去做，问题等价于**s1** 的每种字符的出现次数与**s2**某个子串的每种字符的出现次数相同。 

使用一个长度和 s1 长度相等的固定窗口大小的滑动窗口，在 s2 上面从左向右滑动，判断 s2 在滑动窗口内的每个字符出现的个数是否跟 s1 每个字符出现次数完全相等。利用Counter进行字符统计，窗口右移时将新加入窗口的字符个数在counter2中加1，比较是否相等，再继续右移把左边移出窗口的字符的个数减1，如果个数为0即从字典中删除，left和right都继续向前。

```python
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        # 统计s1的字符个数
        counter1 = collections.Counter(s1)
        # 定义滑动窗口的范围
        left = 0
        right = len(s1) - 1
        # 统计窗口s2[left, right - 1]内的元素出现的次数
        counter2 = collections.Counter(s2[0:right])
        while right < len(s2):
            # right位置元素加入s2中
            counter2[s2[right]] += 1
            # 如果滑动窗口内各个元素出现的次数跟 s1 的元素出现次数完全一致，返回 True
            if counter1 == counter2:
                return True
            # 窗口继续向右移动，把当前 left 位置的元素出现次数 - 1
            counter2[s2[left]] -= 1
            # 如果当前 left 位置的元素出现次数为 0， 需要从字典中删除，
            # 否则这个出现次数为 0 的元素会影响两 counter 之间的比较
            if counter2[s2[left]] == 0:
                del counter2[s2[left]]
            # 窗口继续向右移动
            left += 1
            right += 1
        return False
```

#### 29.尽可能使字符串相等(1208-Medium)

##### Key:问题转换不大于maxCost的最长子数组-滑动窗口

题意：给你两个长度相同的字符串，s 和 t。将 s 中的第 i 个字符变到 t 中的第 i 个字符需要 |s[i] - t[i]| 的开销（开销可能为 0），也就是两个字符的 ASCII 码值的差的绝对值。用于变更字符串的最大预算是 maxCost。在转化字符串时，总开销应当小于等于该预算，这也意味着字符串的转化可能是不完全的。如果你可以将 s 的子字符串转化为它在 t 中对应的子字符串，则返回可以转化的最大长度。如果 s 中没有子字符串可以转化成 t 中对应的子字符串，则返回 0。

分析：计算每个字符对应的开销数值构成开销数组，ord()计算字符的ASCII码，问题转换为求开销数组中和不大于maxCost的最长子数组，利用滑动窗口求解，移动左指针时判断sum和maxCost，最后返回窗口长度。

```python
class Solution(object):
    def equalSubstring(self, s, t, maxCost):
        """
        :type s: str
        :type t: str
        :type maxCost: int
        :rtype: int
        """
        # 问题转换为已知一个数组 costs ，求：和不超过 maxCost 时最长的子数组的长度。
        N = len(s)
        left,right = 0,0
        res = 0
        sum = 0
        costs = [0] * N
        # 计算每个字符转换对应的开销数组
        for i in range(N):
            costs[i] = abs(ord(s[i]) - ord(t[i]))
        while right < N:
            # 将当前窗口的开销累加到sum中
            sum += costs[right]
            # 判断是否超过最大值，超过则移动窗口
            while sum > maxCost:
                sum -= costs[left]
                left += 1
            res = max(res, right - left + 1)
            right += 1
        return res

```

#### 30.数据流中的第K大元素(703-Easy)

##### Key：第K大元素、堆

题意：设计一个找到数据流中第 k 大元素的类（class）。注意是排序后的第 k 大元素，不是第 k 个不同的元素。请实现 KthLargest 类：

KthLargest(int k, int[] nums) 使用整数 k 和整数流 nums 初始化对象。
int add(int val) 将 val 插入数据流 nums 后，返回当前数据流中第 k 大的元素。

分析：常见的面试题，掌握利用堆的自动排序来解题，保持堆内不超过k个元素，超出弹出堆顶元素最小值，最后留下的堆顶元素就是第K大元素。

```python
class KthLargest(object):

    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self.k = k
        self.L = nums
        # 初始化最小堆
        heapq.heapify(self.L)

    def add(self, val):
            """
        :type val: int
        :rtype: int
        """
        heapq.heappush(self.L, val)
        # 保证堆中的元素个数不超过 K个，超过就pop出堆顶元素
        while len(self.L) > self.k:
            heapq.heappop(self.L)
        # 此时堆中的最小元素（堆顶）就是整个数据流中的第 K大元素。
        return self.L[0]
```

#### 31.杨辉三角 II

##### Key: 动态规划滚动数组

题意：给定一个非负索引 *k*，其中 *k* ≤ 33，返回杨辉三角的第 *k* 行。

分析：如果不考虑优化空间复杂度的话，可以利用前一行计算每一行的杨辉三角值，再返回最后一行，和杨辉三角一的做法差不多，如果考虑优化空间复杂度的话，可以使用动态规划中的滚动数组，具体可以结合代码看别人画的示例图，感觉还算挺清晰，关键在于`使用一维数组，内层循环从右向左遍历每个位置，每个位置的元素 dp[j] = dp[j] + dp[j-1]。`![image.png](https://pic.leetcode-cn.com/1613063846-QBYcBY-image.png)

```python
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        result = []
        # [0,rowIndex]
        for i in range(rowIndex+1):
            # 每行生成对应个数全为1的列表
            now = [1] * (i+1)
            for n in range(1, i):
                # 当前位置等于前一行当前位置-1元素值+前一行当前位置值
                now[n] = pre[n-1] + pre[n]
            result = now
            pre = now
        return result

        # 动态规划滚动数组：优化空间复杂度
        # 使用一维数组，然后从右向左遍历每个位置，每个位置的元素 res[j] += 其左边的元素 res[j−1]。
        """
        dp = [1] * (rowIndex + 1)
        for i in range(2, rowIndex + 1):
            for j in range(i-1, 0, -1):
                dp[j] = dp[j] + dp[j-1]
        return dp
        """
```

#### 32.找到所有数组中消失的数字(448-Easy)

题意：给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的整型数组，数组中的元素一些出现了两次，另一些只出现一次。找到所有在 [1, n] 范围之间没有出现在数组中的数字。您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。

分析：将所有正数作为数组下标，置对应数组值为负值。那么，仍为正数的位置即为（未出现过）消失的数字。

```python
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for num in nums:
            # 作为下标需要-1，将对应位置数变为负数
            nums[abs(num)-1] = -abs(nums[abs(num)-1])
            # 还有正数说明对应位置的数字缺失，位置到数字需要+1
        return [i+1 for i,num in enumerate(nums) if num>0]
        # 简易版：利用set去重
        """
        # 构造全部数字的set，减去原set差即为缺失的数字
        return set([i for i in range(1,len(nums)+1)])-set(nums)
        """
```

#### 33.最大连续1的个数(485-Easy)

题意：给定一个二进制数组， 计算其中最大连续1的个数。

分析：数组一次遍历，遍历遇到1记录当前连续1的个数，遇到非1更新最大连续1的个数。如果用滑动窗口，利用idnex记录0的位置，i-index表示当前连续1的个数。

```python
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 第一种方法：数组一次遍历官方题解
        count = 0
        maxNum = 0
        for num in nums:
            if num == 1:
                # 记录当前连续1的个数
                count += 1
            else:
                # 使用之前连续1的个数更新最大连续1的更熟，将当前连续1的个数清零。
                maxNum = max(maxNum, count)
                count = 0
        # 数组的最后一个元素可能是1，最长连续1的子数组可能出现在数组末尾，所以遍历完需要再更新一遍
        maxNum = max(maxNum, count)
        return maxNum

        # 第二种方法滑动窗口，记录遇到0的元素位置
        """
        N = len(nums)
        res = 0
        index = -1
        # 利用index记录0出现的位置
        for i, num in enumerate(nums):
            if num == 0:
                index = i
            else:
                res = max(res, i - index)
        return res
        """
```

#### 34.数组拆分I(561-Easy)

题意：给定长度为 2n 的整数数组 nums ，你的任务是将这些数分成 n 对, 例如 (a1, b1), (a2, b2), ..., (an, bn) ，使得从 1 到 n 的 min(ai, bi) 总和最大，返回该最大总和。

分析：翻译一下题目就是把输入的数组拆成 n*n* 对，将每一对的最小值求和，得到的结果最大。其实就是小数字组成一对、中数字组成一队，大数字组成一对，每对取 min之后，求和得到的结果才是最大的。思路就是对输入的数组nums进行**排序**，然后依次求相邻的两个元素的最小值，总和就是结果。

```python
class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 数组排序，两两组合，取每对前一个累加即和最大
        res = 0
        nums.sort()
        for i in range(0,len(nums),2):
            res += nums[i]
        return res
        # 切片方法
        """
        nums.sort()
        # nums[::2]取偶数下标的数值：0,2,4,……
        return sum(nums[::2])
        """
```

#### 35.重塑矩阵（566-Easy）

题意：在MATLAB中，有一个非常有用的函数 reshape，它可以将一个矩阵重塑为另一个大小不同的新矩阵，但保留其原始数据。给出一个由二维数组表示的矩阵，以及两个正整数r和c，分别表示想要的重构的矩阵的行数和列数。重构后的矩阵需要将原始矩阵的所有元素以相同的行遍历顺序填充。如果具有给定参数的reshape操作是可行且合理的，则输出新的重塑矩阵；否则，输出原始矩阵。

分析：矩阵个数不等于r*c无法转换返回原数组，可以转换新建一个r行c列数组，按行遍历原数组每个位置，利用row和col保存在新数组中，遍历新位置col += 1，col == c到了新数组的右边界需要换行。

```python
class Solution(object):
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        # 获取原数组行列数
        M,N = len(nums), len(nums[0])
        # 无法重塑的情况下返回原数组
        if M * N != r * c:
            return nums
        row,col = 0,0
        # 新建r行c列值为0的结果数组
        res = [[0]*c for _ in range(r)]
        for i in range(M):
            for j in range(N):
                # 到达最后一列，新起一行
                if col == c:
                    row += 1
                    col = 0
                # 逐行遍历放置到对应位置
                res[row][col] = nums[i][j]
                col += 1
        return res
        # numpy的reshape方法
        """
        import numpy
        return np.asarray(nums).reshape((r, c))
        """        
```

#### 36.最大连续1的个数III（1004-Medium）

##### Key:滑动窗口

题意：给定一个由若干 `0` 和 `1` 组成的数组 `A`，我们最多可以将 `K` 个值从 0 变成 1 。返回仅包含 1 的最长（连续）子数组的长度。

分析：题意转换。把「最多可以把 K 个 0 变成 1，求仅包含 1 的最长子数组的长度」转换为 「找出一个最长的子数组，该子数组内最多允许有 K 个 0 」，求最大连续子区间使用滑动窗口方法，限制条件是窗口内最多有K个0，right主动右移，left被动右移。

```python
class Solution(object):
    def longestOnes(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        left,right = 0,0
        res = 0
        zeros = 0
        N = len(A)
        while right < N:
            if A[right] == 0:
                zeros += 1
            # 记录窗口内0的个数与K比较，大于K则移动左指针，同时判断最左元素是否为0
            while zeros > K:
                if A[left] == 0:
                    zeros -= 1
                left += 1
            res = max(res, right - left + 1)
            right += 1
        return res
```

#### 37.K连续位的最小翻转次数(995-Hard)

##### Key：差分数组，滑动窗口

题意：在仅包含 0 和 1 的数组 A 中，一次 K 位翻转包括选择一个长度为 K 的（连续）子数组，同时将子数组中的每个 0 更改为 1，而每个 1 更改为 0。返回所需的 K 位翻转的最小次数，以便数组没有值为 0 的元素。如果不可能，返回 -1。

分析：两个方法-差分数组和滑动窗口，不愧是困难题真心不会。

(1)差分数组：核心数组记录的是在某个点的高度变化值，是一种上下台阶的思想。`diff[i] = A[i-1] - A[i]`，翻转区间只会影响差分值，左区间值加1，后面的数会跟随累加该值，累加到超出右边界需要把超出减掉diff[left]+1,diff[right+1]-1。本题累加差分数组就是当前位置需要翻转的次数，好了多了我感觉也解释不清楚了。

(2)滑动窗口：位置i现在的状态，和它被前面K - 1个元素翻转的次数（奇偶性）有关。使用队列模拟滑动窗口，该滑动窗口的含义是前面 K - 1 个元素中，以哪些位置起始的子区间进行了翻转。队列中元素的个数代表了 i 被前面 K - 1 个元素翻转的次数。原先是0被翻转偶数次还是0仍需要翻转，原先是1被翻转基数次扔需要翻转，所以是否要翻转的判断条件是`len(que) % 2 == A[i]`，需要翻转就把该位置存储在队列中，表示以该位置为起始的K个子区间进行了翻转。

```python
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
```

#### 38.托普利茨矩阵(766-Easy)

题意：给你一个 m x n 的矩阵 matrix 。如果这个矩阵是托普利茨矩阵，返回 true ；否则，返回 false 。如果矩阵上每一条由左上到右下的对角线上的元素都相同，那么这个矩阵是托普利茨矩阵。

分析：每一个位置都要跟其右下角的元素相等，我用了切片操作，第 i 行的 [0,N−2]的切片等于第 i + 1 行的[1,N−1]。两个切片操作：[:-1]去除最后一位，[1:]从第二位开始。

```python
class Solution(object):
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        # 思路：每一个位置都要跟其右下角的元素相等，相邻行列表错位相等
        for i in range(len(matrix)-1):
            # 切片操作：[:-1]去除最后一位；[1:]从第二位开始，i行和i+1行错位列表相等
                if matrix[i][:-1] != matrix[i+1][1:]:
                    return False
        return True
```

#### 39.翻转图像（832-Easy)

题意：给定一个二进制矩阵 A，我们想先水平翻转图像，然后反转图像并返回结果。水平翻转图片就是将图片的每一行都进行翻转，即逆序。例如，水平翻转 [1, 1, 0] 的结果是 [0, 1, 1]。反转图片的意思是图片中的 0 全部被 1 替换， 1 全部被 0 替换。例如，反转 [0, 1, 1] 的结果是 [1, 0, 0]。

分析：可以按照题意直接写，水平翻转就是逆序，可以用切片也可以用`reverse()`函数，会减少空间复杂度，反转图片0和1互换可以用和1异或操作。

```python
class Solution(object):
    def flipAndInvertImage(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        rows = len(A)
        cols = len(A[0])
        for row in range(rows):
            A[row] = A[row][::-1]
            # 或者 A[row] = A[row].reverse()
            for col in range(cols):
                A[row][col] ^= 1
        return A
```

#### 40.转置矩阵（867-Easy）

题意：给你一个二维整数数组 `matrix`， 返回 `matrix` 的 **转置矩阵** 。矩阵的 **转置** 是指将矩阵的主对角线翻转，交换矩阵的行索引与列索引。

分析：注意行列可能不均等，所以需要新建一个N行M列的矩阵来放置元素。

```python
class Solution(object):
    def transpose(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
    # 本题的矩阵的行列数可能不等，因此不能做原地操作，需要新建数组
    # 获取矩阵的行和列
    M,N = len(matrix),len(matrix[0])
    # 新建一个N行M列的矩阵
    res = [[0]*M for i in range(N)]
    for i in range(M):
        for j in range(N):
            res[j][i] = matrix[i][j]
    return res
```

#### 41.单调数列（896-Easy）

题意：如果数组是单调递增或单调递减的，那么它是单调的。如果对于所有 i <= j，A[i] <= A[j]，那么数组 A 是单调递增的。 如果对于所有 i <= j，A[i]> = A[j]，那么数组 A 是单调递减的。当给定的数组 A 是单调数组时返回 true，否则返回 false。

分析：一次遍历，利用标记数组记录是否单调上升或者单调下降，初始设置为True，如果出现上升和下降就对应修改为False。如果数组是单调的则两个标识至少有一个会一直保持True，如果两个同时为False说明数列中既有递增也有递减的情况，非单调数列。

```python
class Solution(object):
    def isMonotonic(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        # 利用两个标识来判定是否递增或者递减
        N = len(A)
        inc, dec = True,True
        for i in range(1, N):
            if A[i] < A[i-1]:
                inc = False
            if A[i] > A[i-1]:
                dec = False
            # 如果inc和dec均为False,说明数列中存在既有递增又有递减的情况，则返回False，如果数组是单调则必有一个一直为True
            if not inc and not dec:
                return False
        return True
```

#### 42.区域和检索-数组不可变

##### Key:前缀和

题意：给定一个整数数组`nums`，求出数组从索引 `i` 到 `j` (i ≤ j）范围内元素的总和，包含 `i`、`j `两点。

分析：应用前缀和的思想，快速计算指定区间段元素之和。preSum[i] 表示i位置左边的元素之和，计算公式为`sum(i, j) = preSum[j + 1] - preSum[i]`

```python
class NumArray(object):
    
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        N = len(nums)
        self.preSum = [0] * (N+1)
        # 计算前缀和数组
        for i in range(N):
            self.preSum[i+1] = self.preSum[i] + nums[i]
    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.preSum[j+1] - self.preSum[i]
```

#### 43.二维区域和检索 - 矩阵不可变(304-Medium)

##### Key：前缀矩阵（二维数组前缀和）

题意：给定一个二维矩阵，计算其子矩形范围内元素的总和，该子矩阵的左上角为 `(row1, col1)` ，右下角为 `(row2, col2)` 。

分析：首先在初始化中列出二维矩阵的前缀矩阵，其次利用前缀矩阵求解子矩阵面积。这里复制二维矩阵有些问题，所以看具体前缀矩阵的求解去看笔记好了。

```python
#  前缀二维矩阵：preSum[i][j]preSum[i][j] 表示 从 [0,0][0,0] 位置到 [i,j][i,j] 位置的子矩形所有元素之和。
class NumMatrix(object):
    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        if not matrix or not matrix[0]:
            M,N = 0,0
        else:
            M = len(matrix)
            N = len(matrix[0])
        # 初始化前缀矩阵，比原矩阵多一行一列：M+1行N+1列
        self.preSum = [[0]*(N+1) for _ in range(M+1)]
        # 前缀矩阵求解公式：preSum[i][j]=preSum[i−1][j]+preSum[i][j−1]−preSum[i−1][j−1]+matrix[i][j]
        for i in range(M):
            for j in range(N):
                self.preSum[i+1][j+1] = self.preSum[i][j+1] + self.preSum[i+1][j] - self.preSum[i][j] + matrix[i][j]

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        # 利用前缀矩阵求子矩形面积
        return self.preSum[row2+1][col2+1] - self.preSum[row2+1][col1] - self.preSum[row1][col2+1] + self.preSum[row1][col1]
        
```

#### 44.俄罗斯套娃信封问题(354-Hard)

##### Key：动态规划

题意：给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。注意：不允许旋转信封。

分析：动态规划中等题，最长上升子序列问题，定义状态dp[i]表示以i结尾的最长递增子序列的长度，用 j 遍历 前i-1个物品，满足条件即更新dp[i]，因为第i件物品是一定要选的，所以状态转移方程为：`dp[i] = max(dp[i], dp[j] + 1)`，结果即为dp中的最大值即为满足条件的最长子序列长度。注意本题信封排序规则应为：按照第一维升序，第一维相同按照第二维降序，w值相同的信封只能选取一个，那么套娃规则一定选h最大的，所以第一维升序完的情况下w值相同要按照第二维降序。

```python
class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        # dp[i] 表示以 i 结尾的最长递增子序列的长度
        if not envelopes:
            return 0
        N = len(envelopes)
        # 按照先第一维升序，一维相同按第二维降序排列
        envelopes.sort(key=lambda x:(x[0], -x[1]))
        # 存储以 i 结尾最长递增子序列的长度，初始化只有一个信封
        dp = [1] * N
        for i in range(N):
            for j in range(i):
                # 比较第二维度，满足情况更新状态转移方程
                if envelopes[j][1] < envelopes [i][1]:
                    # 第i个信封是必须选择的，前i-1的信封有符合条件的就要更新dp[i]
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

#### 45.只出现一次的数字（136-Easy）

题意：给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。说明：你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

分析：不借助额外空间采用异或操作，0和任何数异或结果为任何数，任何数自身异或结果为0，其他数均出现两次，所以最后连续异或的结果为出现一次的数字。

```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 第一种方法：利用异或操作：没有使用额外空间【题目要求】
        # 0和任何数做异或运算结果仍是原来的数，自身与自身异或是0，所以连续异或下来最后剩余的数即为出现一次
        a = 0
        for num in nums:
            a = a ^ num
        return a

        # 第二种方法：使用额外空间，对nums计数   
        """                       
        countMap = collections.Counter(nums)
        N = len(nums)
        for i in range(N):
            if countMap[nums[i]] == 1:
                return nums[i]
         """ 
```

#### 46.用栈实现队列(232-Easy)

题意：请你仅使用两个栈实现先入先出队列。队列应当支持一般队列的支持的所有操作（push将元素x推到队列末尾、pop从队列开头移除并返回元素、peek返回队列开头元素、empty队列为空返回True非空返回False）

分析：利用两个栈（python中即为两个列表）实现队列，元素先入一栈，再出一栈入二栈，二栈出栈即为队列出队顺序。注意要在二栈为空，一栈存在元素的情况下操作pop和peek。

```python
class MyQueue(object):
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack1 = []
        self.stack2 = []


    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: None
        """
        self.stack1.append(x)


    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        # 二栈为空，一栈存在元素的情况下
        # 先入一栈，再出一栈入二栈，再出二栈即为队列出队顺序
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()


    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        # 返回stack2最后一个元素即为队首元素
        return self.stack2[-1]


    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return not self.stack1 and not self.stack2

```

#### 47.比特位计数（338-Medium）

##### Key：二进制中1的个数奇偶数规律关系

题意：给定一个非负整数 **num**。对于 **0 ≤ i ≤ num** 范围中的每个数字 **i** ，计算其二进制数中的 1 的数目并将它们作为数组返回。进阶要求算法复杂度为O(n)。

分析：如果没有算法复杂度的要求可以直接按照题意计算，遍历num每个数先用bin()方法转换成二进制，再count('1')加入结果数组即可获得。有复杂度的要求可以将数按照奇偶分开，建立二进制1个数结果数组，下标即为要判断的数，寻找规律和前后关系：奇数的二进制中1的个数比前一个偶数多末尾的1，偶数二进制中1的个数和其除2后二进制中1的个数相同，因为偶数二进制末尾为0，除2相当于右移不影响1的个数（4:100,2:10,1的个数相同)。

```python
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        # 第一种分奇数和偶数处理：O(n)复杂度【题目要求】
        bits = [0] * (num+1)
        for i in range(1, num+1):
            # 奇数二进制表示一定比前面那个偶数多一个 1
            if i % 2 == 1:
                bits[i] = bits[i-1] + 1
            # 偶数i二进制1的位数与 i/2的二进制1的位数相等（4:100,2:10,1的个数相同）
            # 偶数二进制末尾是0，除2相当于右移一位，1的个数没有变化
            else:
                bits[i] = bits[i//2]
        return bits
        # 第二种方法：直接转换二进制数1的个数，O(n*sizeof(integer))复杂度
        # res = []
        # for i in range(num+1):
        #     res.append(bin(i).count('1'))
        # return res
```

#### 48.回文子串(647-Medium)

##### Key：可用动态规划

题意：给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

分析：动态规划定义状态:dp[j]表示从j到当前遍历的i的字符串是否为回文子串，长度为N初始化为1（是回文子串），计数器初始化为N（因为每个字符都为一个回文串），如果当前字符串首尾相等且子串是回文串时，累加到结果且设置当前位置j的dp为1，不满足条件设置为0。另一种方法为以中心为扩散判断，区分奇偶长度，奇数长度中心为某一个元素，偶数长度中心为某对元素。

```python
class Solution(object):
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # dp[j]表示从j到当前遍历的i的字符串是否为回文子串
        N = len(s)
        # 初始化状态数组为 1，cnt为 N，因为每个单词自身均为一个回文串
        dp = [1] * N
        cnt = N
        for i in range(N):
            for j in range(i):
                # 如果当前字符串首尾相等且子串是回文串，j到当前i的字符串才为回文串，并计数
                if s[j] == s[i] and dp[j+1] == 1:
                    dp[j] = 1
                    cnt += 1
                # 非回文串设置为0
                else:
                    dp[j] = 0
        return cnt

        # 第二种方法：以中心扩散判断
        """
        L = len(s)
        cnt = 0
        # 以某一个元素为中心的奇数长度的回文串的情况
        for center in range(L):
            left = right = center
            while left >= 0 and right < L and s[left] == s[right]:
                cnt += 1
                left -= 1
                right += 1
        # 以某对元素为中心的偶数长度的回文串的情况
        for left in range(L - 1):
            right = left + 1
            while left >= 0 and right < L and s[left] == s[right]:
                cnt += 1
                left -= 1
                right += 1
        return cnt
        """  
```

#### 50.回文链表（234-Easy）

题意：请判断一个链表是否为回文链表。

分析：可以直接将链表存入一个列表后，列表逆序与原列表比较是否相等。采取栈的方法先将链表元素入栈，然后从表头元素开始和栈顶元素依次比较判断是否相等，再向内比较。

```python
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        # 第一种方法：利用栈判断牺牲空间复杂度，先将链表入栈再从表头和栈顶比较
        # 边界条件判断
        if not head and head.next:
            return True
        stack = []
        # 将所有元素入栈
        cur = head
        while cur:
            stack.append(cur.val)
            cur = cur.next
        # 表头元素与栈顶元素比较，相等则出栈继续向内判断
        cur = head
        while stack:
            if cur.val != stack.pop():
                return False
            else:
                cur = cur.next
        return True
        
        # 第二种方法：直接使用列表逆序判断
        """
        res = []
        while head:
            res.append(head.val)
            head = head.next
        return res == res[::-1]
        """

```

#### 51.分割回文串(131-Medium)

##### Key：回溯法

题意：给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。返回 s 所有可能的分割方案。

分析：题意为把输入字符串分割成回文子串的所有可能的结果，求所有可能的结果需要用到回溯法，回溯法是在搜索尝试过程中寻找问题的解，不满足求解条件时就回溯返回尝试别的路径，整体思路为：搜索每一条路，每次回溯是对具体的一条路径而言的。对当前搜索路径下的的未探索区域进行搜索，则可能有两种情况：

（1）当前未搜索区域满足结束条件，则保存当前路径并退出当前搜索；
（2）当前未搜索区域需要继续搜索，则遍历当前所有可能的选择：如果该选择符合要求，则把当前选择加入当前的搜索路径中，并继续搜索新的未探索区域。

本题的**未探索区域**为剩余未搜索的字符串s，**结束条件**为s为空，**未探索区域当前可能的选择**为s的1-length个字符，**当前选择符合要求**是cur是回文字符串，**新的未探索区域**为s去除cur剩余字符串。

```python
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        # 是否为回文串的判断函数
        self.isPalindrome = lambda s:s == s[::-1]
        res = []
        self.backtrack(s,res,[])
        return res
    # 回溯函数：寻找到达结束条件的所有可能路径
    def backtrack(self, s, res, path):
        # 未探索区域满足结束条件
        if not s:
            res.append(path)
            return
        # 切片操作[:i]实际为[0,i-1]，所以 i 要遍历到 len(s) + 1
        # 调试了一下，每次执行到这个循环的时候 path就为空了。
        for i in range(1, len(s) + 1):
            # 当前选择符合回文串条件
            if self.isPalindrome(s[:i]):
                # 递归回溯产生新数组
                self.backtrack(s[i:], res, path + [s[:i]])
```

#### 52.全排列（46-Medium）

##### Key：回溯法

题意：给定一个 **没有重复** 数字的序列，返回其所有可能的全排列。

分析：先写以 1开头的全排列，它们是：[1, 2, 3], [1, 3, 2]，即 1 + [2, 3] 的全排列（注意：递归结构体现在这里）；
再写以 2 开头的全排列，它们是：[2, 1, 3], [2, 3, 1]，即 2 + [1, 3] 的全排列；
最后写以 3 开头的全排列，它们是：[3, 1, 2], [3, 2, 1]，即 3 + [1, 2] 的全排列。
总结搜索的方法：按顺序枚举每一位可能出现的情况，已经选择的数字在当前要选择的数字中不能出现。用回溯的方法求解其他的全排列。

```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        def backtrack(nums,path):
            if not nums:
                res.append(path)
                return
            for i in range(len(nums)):
                # 当前选择的元素和其他元素的全排列组合，不包含当前元素，已经选择的数字在当前要选择的数字中不能出现
                # [1,2,3]：先选出1，和[2,3]的全排列组合，累加到 path
                backtrack(nums[:i] + nums[i+1:],path + [nums[i]])
        backtrack(nums,[])
        return res
```

#### 53.分割回文串II（132-Hard）

##### Key：动态规划

题意：给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是回文。返回符合要求的 **最少分割次数** 。

分析：求分割回文串的最小分割次数类似于最长递增子序列问题，使用动态规划分析本元素和之前元素的关系，当s本身为回文串不做任何判断，不是的时候需要找“刀子j”去分割字符串。状态定义：`dp[i] 是以 i 结尾的分割成回文串的最少次数`状态转移方程：` dp[i] = max(dp[i],dp[j]+1)`。i是当前被回文的子字符串的总长，j是分割的刀子，不断切割字符串为[0:j]和[j+1:i]，更关心[j+1:i]是否构成回文因为动态规划的写法决定了dp[j]已经存储了[0:j]位置的最小的次数，只有子字符串s[j+1:i]是回文字符串的时候，dp[i]可以通过dp[j]加上一个回文字符串s[j+1:i]而得到。注意dp的初始化为N，因为每个字符都可以当做一个回文串，最大分割次数为N，最后结果取dp数组的最后一个数即为以N-1结尾的分割成回文串的最少次数。

```python
class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        # dp初始化为N，最大分割数字是每个字符都可以单独成一个回文
        # dp[i] 是以 i 结尾的分割成回文串的最少次数
        N = len(s)
        dp = [N] * N
        for i in range(N):
            # 0-i本身是一个回文串最小分割次数为0
            if self.isPalindrome(s[:i+1]):
                dp[i] = 0
                continue
            # 0-i不是回文串时候需要用 j 来切割 
            # 只有子字符串 s[j + 1..i] 是回文字符串的时候，dp[i] 可以通过 dp[j] 加上一个回文字符串 s[j + 1..i] 而得到
            for j in range(i):
                if self.isPalindrome(s[j+1:i+1]):
                    dp[i] = min(dp[i],dp[j]+1)
        return dp[-1]
    # 判断是否为回文串
    def isPalindrome(self, s):
        return s == s[::-1]
```

#### 54.最长递增子序列（300-Medium）

##### Key：动态规划

题意：给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

分析：动态规划定义为：`dp[i]以第 i个数字结尾的最长上升子序列的长度`，状态转移方程为：` dp[i] = max(dp[i], dp[j] + 1)`，发生转移的条件是`num[i] > nums[j]`，nums[i]可以加在nums[j]后成为更长的子序列。

更新一下新的方法，因为遇到后面复合题求解最长递增子序列的时候采用动态规划会超时，所以要用贪心+二分的方法：https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/zui-chang-shang-sheng-zi-xu-lie-dong-tai-gui-hua-2/

```python
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        N = len(nums)
        # dp[i]以第 i个数字结尾的最长上升子序列的长度，初始化为 1
        dp = [1] * N
        for i in range(N):
            for j in range(i):
                # nums[i]必须被选取，当nums[i] > nums[j]，表明dp[i]可以从dp[j]的状态转移过来
                # nums[i] 可以放在nums[j] 后面以形成更长的上升子序列
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        # 最后返回 dp 数组中最大值
        return max(dp)
    
 # 贪心+二分查找【高效】
class Solution:
    def lengthOfLIS(self, nums: [int]) -> int:
        tails, res = [0] * len(nums), 0
        for num in nums:
            i, j = 0, res
            while i < j:
                m = (i + j) // 2
                if tails[m] < num: i = m + 1 # 如果要求非严格递增，将此行 '<' 改为 '<=' 即可。
                else: j = m
            tails[i] = num
            if j == res: res += 1
        return res
```

#### 55.删除字符串中的所有相邻重复项(1047-Easy)

题意：给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。在 S 上反复执行重复项删除操作，直到无法继续删除。在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

分析：用栈保存未被消除的所有字符，遍历字符串比较当前字符和栈顶元素是否相同，相同将栈顶元素弹出，不同则该字符入栈。列表转字符串：`''.join(stack)`

```python
class Solution(object):
    def removeDuplicates(self, S):
        """
        :type S: str
        :rtype: str
        """
        stack = []
        N = len(S)
        for i in range(N):
            # 判断栈顶元素和当前字符是否相同，相同则出栈
            if stack and S[i] == stack[-1]:
                stack.pop()
            # 不同则将元素入栈
            else:
                stack.append(S[i])
        # 列表转字符串输出
        return ''.join(stack)
```

#### 56.基本计算器（224-Hard）

题意：实现一个基本的计算器来计算一个简单的字符串表达式 `s` 的值。

分析：逻辑性好强梳理着梳理着就八成透彻吧，主要是用栈保存左边表达式的结果和运算符，多层嵌套的时候保留最里面的嵌套，为了交题而交题。

```python
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 这个题逻辑性太强以至于不想复杂的看下去，over
        res, num, sign = 0, 0, 1
        stack = []
        for c in s:
            if c.isdigit():
                # 为什么num*10
                num = 10 * num + int(c)
            elif c == "+" or c == "-":
                res += sign * num
                num = 0
                sign = 1 if c == "+" else -1
            elif c == "(":
                stack.append(res)
                stack.append(sign)
                res = 0
                sign = 1
            elif c == ")":
                res += sign * num
                num = 0
                res *= stack.pop()
                res += stack.pop()
        # 为什么最后还要加一下呢
        res += sign * num
        return res
```

#### 57.反转链表(206-Easy)

题意：反转一个单链表。

分析：设置三个节点，cur表示当前遍历到的节点，pre表示当前节点的前驱节点，中间变量temp保存当前节点的后驱节点，遍历cur用temp保存cur的后驱节点，cur.next反转指向前驱节点，将pre和cur向后移一位，当cur为Null的时候返回pre。

```python
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # cur表示当前遍历到的节点，pre表示当前节点的前驱节点，需要中间变量temp保存当前节点的后驱节点
        pre = None
        cur = head
        while cur:
            # 先把当前节点的后驱节点保存以免丢失
            temp = cur.next
            # 反转指向前驱节点
            cur.next = pre
            # 将pre，cur往后移一位
            pre = cur
            cur = temp
        # 当cur == Null结束循环返回pre
        return pre
```

#### 58.基本计算器II(227-Medium)

题意：给你一个字符串表达式 `s` ，请你实现一个基本计算器来计算并返回它的值。整数除法仅保留整数部分。

分析：这题没看直接抄

```python
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        a = []
        opmp = {
            "+": lambda e: a.append(e),
            "-": lambda e: a.append(-e),
            "*": lambda e: a.append(e * a.pop()),
            "/": lambda e: a.append(int(operator.truediv(a.pop(), e)))
        }
        op = "+"
        num = 0
        for c in s+"+":
            if c.isdigit():
                num = num * 10 + int(c)
            elif c != " ":
                opmp[op](num)
                op = c
                num = 0
        return sum(a)
```

#### 59.验证二叉树的前序序列化(331-Medium)

##### Key：栈/树的入度和出度

题意：序列化二叉树的一种方法是使用前序遍历。当我们遇到一个非空节点时，我们可以记录下这个节点的值。如果它是一个空节点，我们可以使用一个标记值记录，例如 `#`。

分析：验证输入字符串是否是有效的二叉树的前序序列化，可以用栈或者树的入度出度属性

（1）栈：栈是自底向上符合前序遍历的特点，如何判断一颗子树是否有效，首先考虑如何判断一个节点是叶子节点，当一个节点的两个孩子都是'#'的时候，该节点就是叶子节点，非叶子节点必定有一个孩子非空，技巧：把有效的叶子节点使用'#'代替，此时叶子节点会变成空节点，将元素依次存入栈内，栈内元素大于等于3个元素的时候可以判断是否有叶子节点，如果有的话将其替换成#，再继续循环判断是否还有叶子节点，直到栈内仅剩#一个元素。

（2）计算入度出度，在树中所有节点的入度之和等于出度之和，在一颗二叉树中每个空节点都会提供0个出度和1个入度，每个非空节点会提供2个出度和1个入度，根节点入度为0，遍历一遍字符串，每个节点都累加diff = 出度-入度，遍历到任意一个节点要求diff > 0，因为还没遍历到子节点，出度应该大于入度，当所有节点遍历完成后整棵树的diff == 0，diff初始化为1是因为每加入一个非空节点时，都会对diff先减去1（入度）再加上2（出度），由于根节点没有入度出度为2，因此diff初始化为1为了加入根节点的时候diff-1+2此时diff正好为2。

```python
class Solution(object):
    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """
        # 第一种方法：栈
        # 用栈递归判断是否为树的先序序列，主要判断叶子节点并依次消除
        stack = []
        # 以逗号将序列中分隔开加入栈中
        for node in preorder.split(','):
            # 元素先入栈
            stack.append(node)
            # 循环判断如果出现 '数字 # #'，该node一定是叶子节点，将三个元素出栈，将该节点以#代替继续判断
            while len(stack) >= 3 and stack[-1] == stack[-2] == '#' and stack[-3] != '#':
                # 三元素出栈
                stack = stack[:-3]
                # 用 '#' 代替 ' 数字 # # '
                stack.append('#')
        # 最后栈内只剩元素 # 即为一棵树的先序序列
        return len(stack) == 1 and stack[-1] == '#'

        # 第二种方法：利用树的出入度
        """
        nodes = preorder.split(',')
        diff = 1
        for node in nodes:
            diff -= 1
            if diff < 0:
                return False
            if node != '#':
                diff += 2
        return diff == 0
        """

```

#### 60.设计哈希集合（705-Easy）

题意：不使用任何内建的哈希表库设计一个哈希集合（HashSet），实现 MyHashSet 类：

void add(key) 向哈希集合中插入值 key 。
bool contains(key) 返回哈希集合中是否存在这个值 key 。
void remove(key) 将给定值 key 从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。

分析：用列表实现，添加和清除元素之前先判断是否在集合内部

```python
class MyHashSet(object):
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hashSet = []

    def add(self, key):
        """
        :type key: int
        :rtype: None
        """
        if key not in self.hashSet:
            self.hashSet.append(key)

    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        if key in self.hashSet:
            self.hashSet.remove(key)


    def contains(self, key):
        """
        Returns true if this set contains the specified element
        :type key: int
        :rtype: bool
        """
        if key in self.hashSet:
            return True
        return False
```

#### 61.最长回文子串（5-Medium）

##### Key:二维动态规划

题意：给你一个字符串 `s`，找到 `s` 中最长的回文子串。

分析：采用动态规划设置`dp[i][j]`表示从i到j是否为回文串，是为True，否为False，初始化均为False，边界判断小于2的串必为回文串直接返回，再初始化设置每个字母自身为一个回文串dp矩阵对角线元素均为True，枚举终点和起点，满足`s[i] == s[j]`时长度小于3则设置为True，否则取决于再向中间内部的子串是否为回文串的状态。每次更新完`dp[i][j]`之后要寻找为True的标识，记录最长的回文串长度以及其起始位置，便于切片返回结果。

```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        N = len(s)
        # dp[i][j]表示s[i:j]是否为回文串
        dp = [[False] * N for _ in range(N)]
        start,maxLen = 0,1
        # 边界判断
        if N < 2:
            return s
        # 初始化：每个字母均为独立的回文串
        for i in range(N):
            dp[i][i] = True
        # 枚举终点和起点
        for j in range(1,N):
            for i in range(0,j):
                if s[i] == s[j]:
                    # 小于3则设为回文串
                    if j - i < 3:
                        dp[i][j] = True
                    # 否则取决于子串是否为回文串
                    else:
                        dp[i][j] = dp[i+1][j-1]
                # 更新完dp[i][j]之后，更新回文串max_len和起始位置start
                if dp[i][j]:
                    tempLen = j-i+1
                    if tempLen > maxLen:
                        maxLen = tempLen
                        start = i
        return s[start: start+maxLen]
```

#### 62.设计哈希映射（706-Easy）

题意：不使用任何内建的哈希表库设计一个哈希映射（HashMap）。

实现 MyHashMap 类：MyHashMap() 用空映射初始化对象
void put(int key, int value) 向 HashMap 插入一个键值对 (key, value) 。如果 key 已经存在于映射中，则更新其对应的值 value 。
int get(int key) 返回特定的 key 所映射的 value；如果映射中不包含 key 的映射，返回-1。
void remove(key) 如果映射中存在 key 的映射，则移除 key 和它所对应的 value 。

分析：超大数组用空间换时间，实在是不愿意写数据结构的底层实现。

```python
class MyHashMap(object):
# 超大数组：用空间换时间
    def __init__(self):
        self.data = [-1] * (10**6 + 1)

    def put(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data[key]

    def remove(self, key):
        self.data[key] = -1
```

#### 63.螺旋矩阵（54-Medium）

题意：给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

分析：直接理解题意注意边界的移动，或者利用zip打包函数重新组合矩阵不停读取第一行再pop出去剩余矩阵解压重新打包组合再逆序（相当于逆时针旋转的过程）再接着取第一行，或者直接使用numpy的逆时针旋转函数`np.rot90(matrix,1)`。

```python
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        # 思路：循环【取矩阵第一行，逆时针旋转九十度】
        import numpy as np
        res = []
        while matrix:
            res += matrix.pop(0)  #每次提取第一排元素
            # zip函数+列表逆序：将剩余的元素进行逆时针旋转九十度
            # *matrix先解压，然后再用zip重新组合，逆序
            matrix = list(zip(*matrix))[::-1]   
            # matrix = [(6,9),(5,8),(4,7)]
            # numpy逆时针旋转函数
            # matrix[:]=np.rot90(matrix,1).tolist()
        return res

```

#### 64.螺旋矩阵II

##### Key:矩阵遍历

题意：给你一个正整数 `n` ，生成一个包含 `1` 到 `n2` 所有元素，且元素按顺时针顺序螺旋排列的 `n x n` 正方形矩阵 `matrix` 。

分析：螺旋向内遍历矩阵，遍历的过程中依次放入 1-N^2 等各个数字，矩阵遍历需要考虑四个问题，起始位置；移动方向；边界；结束条件。

（1）起始位置：矩阵的左上角（0,0）；

（2）移动方向：本题的移动方向是右下左上循环反复，利用数组来表示，每次当移动了边界会更改方向，但是边界不是固定的。

（3）边界：本题的边界是随着遍历的过程而变化的，螺旋遍历时候已经遍历的数字不能再次遍历，所以边界应该越来越小，规则是**如果当前行（列）遍历结束之后，就需要把这一行（列）的边界向内移动一格。**用四个变量标记边界，初始时分别指向矩阵的四个边界，例如我们把第一行遍历结束（遍历到了右边界），此时需要修改新的移动方向为向下、并且把上边界 up 下移一格，即从旧 up 位置移动到新 up 位置。

（4）结束条件：螺旋遍历的结束条件是所有的位置都被遍历到

```python
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        if n == 0: return []
        # 初始化n*n结果矩阵
        res = [[0] * n for i in range(n)]
        # 四个方向的边界
        left, right, up, down = 0, n - 1, 0, n - 1
        # 当前位置
        x, y = 0, 0
        # 移动方向：右下左上，移动到了边界更改方向
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # 表示当前的移动方向的下标，dirs[cur_d] 就是下一个方向需要怎么修改 x, y
        cur_d = 0
        count = 0
        # 结束条件count元素个数等于n^2
        while count != n * n:
            # 矩阵按照遍历到的位置赋值
            res[x][y] = count + 1
            count += 1
            # 到达右边界，cur_d更改移动方向由右到下，当前行遍历结束，up边界向内移动一格
            if cur_d == 0 and y == right:
                cur_d += 1
                up += 1
            # 到达下边界，cur_d更改移动方向由下到左，当前列遍历结束，right边界想内移动一格
            elif cur_d == 1 and x == down:
                cur_d += 1
                right -= 1
            # 同理
            elif cur_d == 2 and y == left:
                cur_d += 1
                down -= 1
            elif cur_d == 3 and x == up:
                cur_d += 1
                left += 1
            cur_d %= 4
            # 获取矩阵位置坐标
            x += dirs[cur_d][0]
            y += dirs[cur_d][1]
        return res
```

#### 65.不同的子序列（115-Hard）

##### Key：动态规划

题意：给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）

分析：状态定义：`dp[i][j]为s[:j]的子序列中 t[:i]出现的次数`;状态转移方程：`dp[i][j] = dp[i][j-1] + dp[i-1][j-1]/dp[i][j] = dp[i][j-1] `，主要推导过程可以列出两个字符串画出一个矩阵推导递推关系，第一行为1表示任何字符串必包含空串，第一列为0因为空串不包含任何子串。当前s和t的末尾字符相同的时候，状态转移方程分两种情况，分别是用当前字符的情况`dp[i-1][j-1]`以及不用当前字符的情况`dp[i][j-1]`，不同的时候则直接不考虑当前字符`dp[i][j-1]`。

```python
class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        # 状态定义：dp[i][j] = s[:j]的子序列中 t[:i]出现的次数。
        # 状态转移方程：dp[i][j] = dp[i][j-1] + dp[i-1][j-1]/dp[i][j] = dp[i][j-1] 
        # t[i-1] 与 s[j-1]比较是否相等：不用s[j-1]这个字符或者用s[j-1]这个字符
        m = len(s)
        n = len(t)
        # 构建n+1行m+1列的状态转移数组，加入一行一列空串
        # dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)] 
        dp = [[0]*(m+1) for _ in range(n+1)]
        # 初始化定义一下dp数组，空串是任何字符串的子串
        for j in range(m+1):
            dp[0][j] = 1
        for i in range(1, n+1):
            for j in range(1, m+1):
                if t[i-1] == s[j-1]:        # j位置的状态主要取决于j-1的位置
                    # 两种情况：用s[j-1]这个字符（只考虑i-1和j-1的情况，当前字符已经固定） + 不用当前s[j-1]字符，行不变（t不变）列减1排除当前字符
                    dp[i][j] = dp[i-1][j-1] + dp[i][j-1]
                else:
                    # 不相等的情况直接不用当前字符
                    dp[i][j] = dp[i][j-1]
        # 返回dp矩阵最后的值就是结果
        return dp[-1][-1]
```

#### 66.两数之和（1-Easy）

题意：给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出和为目标值的那两个整数，并返回它们的数组下标。你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。你可以按任意顺序返回答案。

分析：如果按照暴力遍历时间复杂度是O(N^2)，本题的最佳题解应该是利用字典存储列表中元素和对应位置构造键值对关系，这样查找nums2的时候就可以直接去字典中查找降低时间复杂度，在字典中找到nums2的位置直接与当前位置组成列表返回。

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 最佳方法：利用哈希表建立值和位置的对应关系，再查找nums2的时候可以降低时间复杂度
        dict = {}
        # 遍历列表
        for index,nums in enumerate(nums):
            # 寻找nums2是否在字典中，在的话将其作为key找到位置
            nums2 = target - nums
            if nums2 in dict:
                return [dict[nums2],index]
            else:
                # 建立数值和位置构成的字典键值对
                dict[nums] = index
```

#### 67.两数相加（2-Medium）

题意：给你两个非空的链表，表示两个非负的整数。它们每位数字都是按照逆序的方式存储的，并且每个节点只能存储一位数字。请你将两个数相加，并以相同形式返回一个表示和的链表。你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

分析：数字以逆序存储在链表中，得到结果之后也是以逆序链表的形式存储，所以直接对链表逐位相加即可，注意每次存储和记录进位，进位为整除10得到，余下值为除10取余得到，构成新结点连接在结果链表头结点后面，移动指针继续向前，主要是要累加进位，while循环判断进位是否也存在，如果链表都遍历完进位还有剩余的话将进位数值也留下链到后面。

```python
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # 结果列表设置头结点
        res = ListNode(-1)
        # 设置当前指针
        cur = res
        # 向前进位
        carry = 0
        # 注意如果遍历完进位还存在时要记录进位
        while(l1 or l2  or carry):
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            # 逐位相加，carry为上一个的进位
            s = carry + x + y
            # 获取当前进位
            carry = s // 10
            # 记录当前的值
            cur.next = ListNode(s%10)
            # 移动指针
            cur = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return res.next
```

#### 68.反转链表II（92-Meidum）

题意：给你单链表的头节点 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。

分析：三种方法，第一种以反转链表206为基础，先从原链表中切出待翻转子链表，调用函数翻转之后再连接回原链表；第二种方法利用栈的数据结构存储待翻转结点位置的值，二次遍历链表再将对应位置的值pop出去实现交换的目的；第三种方法官方题解法二，利用头插法每次遍历到的结点都插入翻转部分的起始位置，当前指针和后继指针间接向后移动再将后续结点依次插入到起始位置实现翻转。

```python
class Solution(object):
    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        # 设置单链表反转函数：给出头结点链表反转
        def reverseLinkNode(head):
            pre = None
            cur = head
            while cur:
                temp = cur.next
                cur.next = pre
                pre = cur
                cur = temp
        # 设置虚拟头结点以防头结点发生变化
        dummpy_node = ListNode(-1)
        dummpy_node.next = head
        pre = dummpy_node
        # 第一步：找到原链表中左边位置结点的前驱pre和右边位置结点right_node
        for _ in range(left-1):
            pre = pre.next
        right_node = pre
        for _ in range(right - left + 1):
            right_node = right_node.next
        
        # 第二步：将子链表切除出来，保留右节点的后继
        left_node = pre.next
        succ = right_node.next
        # 切断连接
        pre.next = None
        right_node.next = None
        # 反转子链表，反转之后的链表头结点right_node，尾结点为left_node
        reverseLinkNode(left_node)
        # 将反转好的子链表和前后连接起来
        pre.next = right_node
        left_node.next = succ

        return dummpy_node.next


        # 第二种方法：利用栈存储待翻转的子链表
        """
        p = head
        stack = []
        # 判断当前遍历位置是否在left和right内部
        loc = 1
        while p :
            # 将子链表区间内的值入栈
            if loc in [left,right]:
                stack.append(p.val)
            p = p.next   
            loc += 1
        # 二次遍历出栈替换链表原位置的值
        res = head
        loca = 1
        while res:
            if loca in [left,right]:
                res.val = stack.pop()
            res = res.next
            loca += 1
        return head
        """

        # 第三种方法：一次遍历头插法
        # 在需要反转的区间里，每遍历到一个节点，让这个新节点来到反转部分的起始位置
        """
        dummpy_node = ListNode(-1)
        dummpy_node.next = head
        pre = dummpy_node
        # pre:永远指向待翻转区域第一个结点的前驱
        for _ in range(left-1):
            pre = pre.next
        # 指向待翻转区域的第一个结点
        cur = pre.next
        for _ in range(right - left):
            # next永远指向 curr 的下一个节点，随着cur的变化而变化
            next = cur.next
            cur.next = next.next
            next.next = pre.next
            pre.next = next
        return dummpy_node.next
        """
```

#### 69.设计停车系统（1603-Easy）

题意：请你给一个停车场设计一个停车系统。停车场总共有三种不同大小的车位：大，中和小，每种尺寸分别有固定数目的车位。

请你实现 ParkingSystem 类：ParkingSystem(int big, int medium, int small) 初始化 ParkingSystem 类，三个参数分别对应每种停车位的数目。bool addCar(int carType) 检查是否有 carType 对应的停车位。 carType 有三种类型：大，中，小，分别用数字 1， 2 和 3 表示。一辆车只能停在  carType 对应尺寸的停车位中。如果没有空车位，请返回 false ，否则将该车停入车位并返回 true 。

分析：利用数组来存储空闲停车位数量，0来补位，下标对应相应的车型。

```python
class ParkingSystem(object):

    def __init__(self, big, medium, small):
        """
        :type big: int
        :type medium: int
        :type small: int
        """
        # 利用数组保存剩余车位书目，下标获取对应车型停车位数目
        self.parkArea = [0, big, medium, small]


    def addCar(self, carType):
        """
        :type carType: int
        :rtype: bool
        """
        # 利用车型索引到空余车位不为0的情况就停车进去，车位减一
        if self.parkArea[carType]:
            self.parkArea[carType] -= 1
            return True
        return False
```

#### 70.无重复字符的最长子串（3-Meidum）

##### Key：滑动窗口

题意：给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

分析：利用滑动窗口，维护一个队列，遇到重复字符不断移动左指针直到将其移除，再将新元素加入队列中，左右指针不断计算当前子串的长度并更新最大值。

```python
import collections
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 维护一个列表队列/集合，遇到重复元素则向右移动窗口直至将该元素移除，再添加新元素
        left, res = 0,0
        c = []
        # c = set()
        for right,val in enumerate(s):
            # 判断遍历到的字符是否已经存在于窗口中，存在则向右移动窗口，直到将该重复元素移除窗口
            while val in c:
                c.remove(s[left])
                left += 1
            c.append(val)
            # c.add(val)
            res = max(res, right - left + 1)
        return res
```

#### 71.逆波兰表达式求值（150-Medium）

题意：根据逆波兰表示法，求表达式的值。有效的算符包括 `+`、`-`、`*`、`/` 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

分析：逆波兰表达式即为后缀表达式，即操作数1，操作数3，运算符2，利用栈数据结构，构建运算符和对应运算的字典结构，遍历tokens表达式，判断在字典中即弹出栈内两个操作数进行运算（除法注意先弹出的是除数，后弹出的是被除数），不在字典则入栈（注意将字符转换为数字），最后栈内即为表达式的结果。

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        # 设置操作符字典，每个操作符对应不同的运算操作
        operators = {'+':operator.add, '-':operator.sub,
                     '*':operator.mul, '/':lambda a,b:int(a/b)}
        stack = []
        for c in tokens:
            # 遇到操作符则弹出栈顶两个数字进行运算，将运算结果入栈
            if c in operators:
                a = stack.pop()
                b = stack.pop()
                # 注意先弹出的是除数，后弹出的是被除数
                stack.append(operators[c](b,a))
            # 非操作符转换成数字则直接入栈
            else:
                stack.append(int(c))
        # 最后栈内元素即为表达式最后的结果
        return stack[-1]
```

#### 72.合并两个有序链表（21-Easy）

题意：将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

分析：两种方法，可以进行迭代，新建头结点，判断两个链表中结点的值比较大小选择链接到新表头的后面，如果两个链表还有哪个链表有剩余部分的话再链接到结果链表的后面。

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 迭代方法
        new_head = ListNode(-1)
        cur = new_head
        # 依次比较两个链表的结点值，将小的值连接到新链表后
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        # l1或l2剩余部分链接到新链表后面
        cur.next = l1 if l1 else l2
        return new_head.next
```

#### 73.括号生成（22-Meidum）

##### Key：回溯法，深度优先遍历DFS

题意：数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

分析：本题在于需要画出树形结构图，括号生成抽象化成树状结构即为一颗满二叉树，首先先考虑利用dfs将所有情况遍历，然后发现有些情况需要剪枝，左括号数大于n或者右括号数大于左括号数都是无效的，当满足有效情况且路径长度为2*n的时候，将该路径加入结果集即可。

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        #  特殊况特判：
        if n <= 0:
            return []
        res = []
        
      # 回溯法深度优先遍历：括号生成问题抽象成一颗树结构：满二叉树遍历，再根据左右括号的数量剪枝
        def dfs(path, left, right):
            if left > n or right > left:
                return
            # 树的一条路径长度等于2*n的时候是有效组合（n对括号共有2*n个括号）
            if len(path) == 2*n:
                res.append(path)
                return
            dfs(path+'(', left+1, right)
            dfs(path+')', left, right+1)

        # 回溯起点
        dfs("",0, 0)
        return res
```

#### 74.矩阵置零（73-Meidum）

题意：给定一个 m*n 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用原地算法，建议使用常量空间O(1)。

分析：原地算法为即在函数的输入矩阵上直接修改，而不是 return 一个矩阵，本题有几种空间复杂度的解决方法，代码里只列出O(m+n)和O(1)的解法，第一种容易想到对于遍历到0的位置标记对应的行和列，二次遍历的时候根据标记位再将对应行列修改为0。第二种主要思想要将这个标记存储在矩阵的第一行和第一列中。

```python
    class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # 空间复杂度O(m+n)的解法：记录每行每列是否出现0
        m = len(matrix)
        n = len(matrix[0])
        # 标记数组记录0位置出现的行列
        row,col = [False] * m,[False] * n

        for i in range(m):
            for j in range(n):
                # 出现0的位置将其行列都标注为True
                if matrix[i][j] == 0:
                    row[i] = col[j] = True
        for i in range(m):
            for j in range(n):
                # 二次遍历将之前标注为True的行列置为0
                if row[i] or col[j]:
                    matrix[i][j] = 0

        # 空间复杂度O(1)的解法：使用第 0 行和第 0 列来保存 matrix[1:M][1:N] 中是否出现了 0
        """
        if not matrix or not matrix[0]:
            return
        row0,col0 = False, False
        M, N = len(matrix),len(matrix[0])
        # 1. 统计第一行和第一列是否有0
        for i in range(M):
            if matrix[i][0] == 0:
                col0 = True
        for j in range(N):
            if matrix[0][j] == 0:
                row0 = True
        # 2.开始遍历矩阵，遇到0标记到对应的第一行和第一列
        for i in range(1, M):
            for j in range(1, N):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        
        # 3.看第一行和第一列哪位元素为0.将对应的行和列元素全部标记位0
        for i in range(1, M):
            for j in range(1, N):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        # 4.最后根据第一行和第一列的标记位判断是否存在0元素，有的话将其行列置为0
        if row0:
            for j in range(N):
                matrix[0][j] = 0
        if col0:
            for i in range(M):
                matrix[i][0] = 0
        """
```

#### 75.二叉树的中序遍历（94-Meidum）

题意：给定一个二叉树的根节点 `root` ，返回它的 **中序** 遍历。

分析：中序遍历的顺序是左结点-根结点-右结点，本题有递归法和迭代法，迭代法利用栈数据结构寻找到当前树的最左下角即将全部左子树直到遍历到空，出栈将对应的值入结果链表，寻找当前出栈结点的右结点继续遍历。

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
    # 迭代法：将根节点以及对应的左子树入栈，找到每棵树的最左下角，直到没有左子树时将栈内元素弹出并寻找其右子树
        stack,res = [], []
        cur = root
        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res
        # 递归法：
        """
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
        """
```

#### 76.位1的个数（191-Easy）

##### Key：位运算

题意：编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。

分析：直接用bin()函数可以得到一个整数的二进制字符串，统计1的个数即可，注意如果题目问的是二进制中0的个数，答案要count('0')之后减1因为二进制字符串是以'0b'开头。采用位运算的方法：`n&(n-1)`可以把n的二进制中最后一个出现的1改写成0，每次执行这个操作就会消掉n的二进制中最后一个出现的1，因此执行`n&(n-1)`使得n变成0的操作次数，就是n的二进制中1的个数。

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        # 位运算
        count = 0 
        while(n):
            # 按位与：将二进制最低位的1变成0
            n &= (n-1)
            count += 1
        return count

        # 一句话解决一道题系列
        # return bin(n).count('1')
```

#### 77.扁平化嵌套列表迭代器（341-Meidum）

##### Key：DFS+Queue / Stack

题意：给你一个嵌套的整型列表。请你设计一个迭代器，使其能够遍历这个整型列表中的所有整数。

列表中的每一项或者为一个整数，或者是另一个列表。其中列表的元素也可能是整数或是其他列表。

分析：本题定义了一个类NestedInterger，可以存储int或者List类型称之为嵌套列表，相当于一颗多叉树，每个节点都可以有很多子节点。有三个方法可以调用，isInteger()判断当前存储对象是否为int，getInteger()如果当前存储元素是int型，返回当前的结果int，否则调用失败。主要两个思路：在构造函数中提前扁平化整个嵌套列表【递归】，在调用hasNext()方法的时候扁平化当前的嵌套子列表【迭代】。迭代方法主要利用栈存储元素，在调用遍历过程中展开子列表，递归在初始化时候利用深度优先搜索将所有子列表展开入队列。

```python
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        # 迭代：在出栈过程中判断是数字或是列表，列表直接展开入栈
        self.stack = []
        for i in range(len(nestedList)-1,-1,-1):
            self.stack.append(nestedList[i])
    
    def next(self) -> int:
        res = self.stack.pop()
        return res.getInteger()
        
    def hasNext(self) -> bool:
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
```

#### 78.根据身高重建队列(406-Meidum)

题意：假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。

给个示例：输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]；输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]

分析：题目的意思是最后结果队列应该是按照每个位置的整数对都是合理的，即每个位置排在它前面的比它高的确实是它的ki，遇见这种整数对还涉及排序，根据第一个元素正向排序第二个元素反向排序或者根据第一个元素反向排序第二个元素正向排序，往往能够简化解题过程。本题最后队列比较重要的在于ki前面比它高的人数，所以我们采用首先对第一个元素降序排序让身高高的在前面，对第二个元素升序排序让ki大的在后面。排序之后遍历列表中每个整数对，判断当前结果集已经站在前面的人数与遍历到每个元素的ki比较，如果当前站在前面的人数小于当前元素应该站在其前面的元素，则该元素正常入结果集，如果大于则需要将当前元素根据其ki插入到正确的位置上。

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
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
```

#### 79.组合总和（39-Meidum）

##### Key：回溯法

题意：给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的数字可以无限制重复被选取。

分析：遇到列出可行解的问题想到回溯法，本题注意可以剪枝的位置，先对候选集排序即可简化当前和大于目标值的时候可以无需再遍历直接return，回溯过程中不断计算累加sum和列表。

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        # 判空
        if not candidates:
            return []
        res = []
        # 先排序，便于挑选过程中遍历到元素累加已经大于target即return无需再继续
        candidates.sort()
        # 回溯方法：注意剪枝
        def backtrace(i,temp_sum,temp_list):
            # 当前累加和满足退出条件，即把当前路径加入结果集
            if temp_sum == target:
                res.append(temp_list)
                return
            # 剪枝：当前累加和大于目标无需再继续回溯
            if temp_sum > target:
                return
            # 选择未探索区域进行新的回溯，累加和累加，累加列表累加
            for j in range(i,len(candidates)):
                backtrace(j,temp_sum+candidates[j],temp_list+[candidates[j]])
        # 入口
        backtrace(0,0,[])
        return res
```

#### 80.132模式（456-Meidum）

##### Key：单调栈

题意：给你一个整数数组 nums ，数组中共有 n 个整数。132 模式的子序列 由三个整数 nums[i]、nums[j] 和 nums[k] 组成，并同时满足：i < j < k 和 nums[i] < nums[k] < nums[j] 。如果 nums 中存在 132 模式的子序列 ，返回 true ；否则，返回 false。

分析：维护132模式中的3,1尽可能小，2尽可能大，遍历的位置j相当于132模式中的3，需要找到3左边的最小元素为1，找到3右边的比3小的最大元素为2。

   首先遍历一次得到求任何位置的左边最小元素，然后从右向左遍历建立单调递减栈，将当前元素nums[j]入栈时需要把栈内比它小的元素全都pop出来，越往栈底越大所以pop出的最后一个元素就是比3小的最大元素。判断1位置当前元素左侧最小元素与刚出栈元素比较，如果小于即可得到一个132模式。

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        # 维护3位置，1位置找3左边的最小元素，2位置找3右边比3小的最大元素
        # 找1：左侧最小值数组 ；找3：右侧单调递减栈 + 从右向左遍历
        N = len(nums)
        # 初始化一个负无穷数组
        leftMinNum = [float("inf")] * N
        stack = []
        # 构建：任何位置左侧最小值数组，为了寻找1位置元素
        for i in range(1, N):
            leftMinNum[i] = min(leftMinNum[i-1], nums[i-1])
        for j in range(N-1,-1,-1):
            numsk = float("-inf")
            # 找到2位置元素：比当前值小的全部出栈，最后pop出的即为比3位置小的最大元素记录下来
            while stack and stack[-1] < nums[j]:
                numsk = stack.pop()
            # 判断2位置元素是否大于1位置元素
            if leftMinNum[j] < numsk:
                return True
            # 栈顶元素大于当前元素，将当前元素入栈构造一个单调递减栈
            stack.append(nums[j])
        return False
```

#### 81.删除排序链表中的重复元素II(82-Medium)

题意：存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中没有重复出现 的数字。返回同样按升序排列的结果链表。

分析：给定的链表是排好序的，所以**重复的元素在链表中出现的位置是连续的**，我们可以进行一次遍历，设置哑结点指向头结点，从第一个结点开始遍历，判断连续结点的值是否相同，如果相同记录下来，从当前出现重复元素的结点开始向后遍历，等于重复值的结点即可删除，最后返回哑结点的next即可。

[今天绝对降低正确率，写了一个栈的方法调试了好久交错了好多次最后还超时了好家伙真气鸭我可]

```python
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # 一次遍历，重复的元素在链表中出现的位置是连续的
        # 建立哑结点指向head
        dummpy = ListNode(0,head)
        cur = dummpy
        # 从第一个结点开始遍历，当连续两个结点存在时判断是否值相同
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                # 记录下重复的值
                x = cur.next.val
                # 从出现重复值的位置开始遍历，后续等于x需要删除
                while cur.next and cur.next.val == x:
                    cur.next = cur.next.next
            else:
                cur = cur.next
        return dummpy.next
```

#### 82.多数元素（169-Easy）

题意：给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。你可以假设数组是非空的，并且给定的数组总是存在多数元素。

分析：首先想到的是利用哈希表存储每个元素出现的次数，然后遍历找到次数大于n/2的数返回即可，官方题解的说法首先题目的意思其实是在找众数，直接遍历哈希表中所有的键，寻找对应的值最大的即可，利用max()函数。或者直接对数组排序，下标为n/2的必然为众数。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 哈希表存储每个元素出现的次数，形成元素-次数键值对
        counter = collections.Counter(nums)
        n = len(nums)
        # 遍历找到对应次数大于n/2的值，返回其对应的键
        for num in nums:
            if counter[num] > int(n/2):
                return num
```

#### 83.删除排序链表中的重复元素（83-Easy）

题意：存在一个按升序排列的链表，给你这个链表的头节点 `head` ，请你删除所有重复的元素，使每个元素 **只出现一次** 。返回同样按升序排列的结果链表。

分析：遍历链表判断当前结点，因为是升序链表重复元素必然相邻，所以出现重复元素就调整指针位置。

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        cur = head
        # 当后续结点存在的时候
        while cur.next:
            # 因为已经是升序链表所以重复元素必相邻，出现重复元素执行删除操作
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            # 未出现指针继续后移
            else:
                cur = cur.next
        return head
```

#### 84.旋转链表（61-Medium）

##### Key：快慢指针

题意：给你一个链表的头节点 `head` ，旋转链表，将链表每个节点向右移动 `k` 个位置。

分析：将链表每个节点向右移动 k 个位置，相当于把链表的后面 `k % len` 个节点移到链表的最前面。（len 为 链表长度），所以本题分为三个步骤，首先求链表长度，其次找出倒数第k+1个结点，最后是链表重整，将链表倒数第k+1个结点和倒数第k个结点断开，把后半部分拼接到链表的头部。

​	**如何找到链表倒数第k个结点**，设置两个快指针，先让快指针指向链表的第k+1个节点，slow指向第1个节点，然后slow和fast同时向后移动，当fast移动到链表的最后一个节点的时候，那么slow指向链表的倒数第k+1个节点，slow.next即为倒数第k个结点，也是本题需要返回链表的新结点。

```python
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        # 思想：将链表每个节点向右移动 k 个位置，相当于把链表的后面 k % len  个节点移到链表的最前面。（len 为 链表长度）
        # 方法：用快慢指针找到链表的后面 k % len  个节点，slow指向这一段前驱，断开原链接设置新头结点，，fast与头结点相连将这段置于前端
        # 特判
        if not head or not head.next:
            return head
        cur = head
        len_listNode = 0
        # 求链表长度
        while cur:
            len_listNode += 1
            cur = cur.next
        # 长度取模
        k %= len_listNode
        if k == 0:
            return head
        # 设置快慢指针，慢指针和快指针相距 K，快指针指向 k+1
        slow, fast = head,head
        for _ in range(k):
            fast = fast.next
        # 快指针指到链表最后一个结点，慢指针slow指向链表倒数第 k+1 结点
        while fast.next:
            fast = fast.next
            slow = slow.next
        # 新链表的开头即为倒数第 k 个结点
        new_head = slow.next
        # 将倒数第 k+1 结点与 倒数第k个结点断开
        slow.next = None
        # 让链表最后一个结点指向头结点
        fast.next = head
        return new_head
```

#### 85.删除链表的倒数第N个结点（19-Meidum）

##### Key：快慢指针

题意：给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

分析：设置快慢指针，设置哑结点指向头结点便于处理当删除第一个元素的时候，slow指针初始指向dummpy哑结点，便于获得待删除元素的前驱，fast指针初识指向head，向后移动n个元素后指向第n+1个元素。快慢指针一起移动，当快指针移动到链表末尾指向None时此时slow指向倒数第n+1个元素即为倒数第n个元素的前驱，执行删除操作即可。纯属于前一个题的趁热打铁。

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # 设置哑结点是便于删除第一个元素
        dummpy = ListNode(0,head)
        slow,fast = dummpy,head
        # fast指向第n+1个结点
        for _ in range(n):
            fast = fast.next
        # fast指向最后为空时，slow指向倒数第n+1个结点（slow从dummpy开始，直到指向待删除结点前驱）
        while fast:
            fast = fast.next
            slow = slow.next
        # slow.next就是第n个结点
        slow.next = slow.next.next
        return dummpy.next
```

#### 86.二叉搜索树迭代器

题意：实现一个二叉搜索树迭代器类BSTIterator ，表示一个按中序遍历二叉搜索树（BST）的迭代器：
BSTIterator(TreeNode root) 初始化 BSTIterator 类的一个对象。BST 的根节点 root 会作为构造函数的一部分给出。指针应初始化为一个不存在于 BST 中的数字，且该数字小于 BST 中的任何元素。boolean hasNext() 如果向指针右侧遍历存在数字，则返回 true ；否则返回 false 。
int next()将指针向右移动，然后返回指针处的数字。
注意，指针初始化为一个不存在于 BST 中的数字，所以对 next() 的首次调用将返回 BST 中的最小元素。你可以假设 next() 调用总是有效的，也就是说，当调用 next() 时，BST 的中序遍历中至少存在一个下一个数字。

分析：在遍历next的过程中寻找下一个结点，根据中序遍历：左结点-根结点-右结点的顺序，用栈存储，初始化过程中将所有左子树入栈，在next遍历中弹出栈顶元素，寻找栈顶元素的右子树，存在的话将其加入栈中，继续寻找其所有左子树。在hasnext中直接判断栈内是否还有元素，有元素即返回True。

```python
class BSTIterator:

    def __init__(self, root: TreeNode):
        # 一路到底，将根节点和它的所有左节点放到栈中
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left


    def next(self) -> int:
        # 弹出栈顶的节点；如果它有右子树，则对右子树一路到底，把它和它的所有左节点放到栈中。
        cur = self.stack.pop()
        node = cur.right
        while node:
            self.stack.append(node)
            node = node.left
        return cur.val


    def hasNext(self) -> bool:
        # 判断栈内是否有元素
        return len(self.stack) > 0
```

#### 87.颠倒二进制位（190-Easy）

##### Key：运算符和二进制

题意：颠倒给定的 32 位无符号整数的二进制位。

分析：res初始化为0，将res不断左移为n的末尾二进制位提供位置，通过`n & 1`方法获取n的二进制末尾位，与左移好的res进行按位或运算将其添加到res二进制当中，n继续右移连续处理。

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        # n & 1:得到的是将 n 转换为二进制后的最后一位
        # " << "：左移运算符，将指定二进制向左移动一位，低位补0
        # " >> "：右移运算符，将制定二进制向右移动一位。
        #" | "：按位或运算符：只要对应的二个二进位有一个为1时，结果位就为1
        res = 0
        for _ in range(32):
            # res左移末尾为 0 ，取n的末尾二进制位，为1则按位或运算res末尾位为1，为0则按位或运算res末尾位为0，res前面其他位不变。
            # 相当于将n的末尾二进制位加到res末尾
            res = (res << 1) | (n & 1)
            # n继续右移
            n >>= 1
        return res
```

#### 88.搜索二维矩阵（74-Medium）

##### Key:二分查找

题意：编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：每行中的整数从左到右按升序排列。每行的第一个整数大于前一行的最后一个整数。

分析：本题题意给出矩阵的性质：每行元素都是单调递增的，并且下一行的元素会比本行更大。先寻找到目标元素所在行，将目标元素与每行最末尾元素比较，假如 `target < matrix[i][N - 1]` 时，说明 target 可能在本行中出现，下面各行的元素都大于上行末尾元素，所以target不可能在下面的行中出现。在本行中可以使用顺序查找或者二分查找。

```python
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        M = len(matrix)
        N = len(matrix[0])
        for i in range(M):
            # 升序性质，将目标元素与每行最后一个元素比较
            if target > matrix[i][N-1]:
                continue
            # 如果目标小于当前行的最大元素，由于下面的行内元素均是大于当前行，所以目标如果存在的话就在当前行内
            if target in matrix[i]:
                return True
        return False
        # 全局二分法：二维矩阵当做一维矩阵，前提每一行最后一个元素小于下一行第一个元素
        """
        M,N = len(matrix),len(matrix[0])
        left, right = 0, N-1
        while left <= right:
            # 根据 mid 求出在二维矩阵中的具体位置
            mid = left + (right-left) //2
            cur = matrix[mid // N][mid % N]
            # 判断 left 和 right 的移动方式
            if cur == target:
                return True
            elif cur < target:
                left = mid + 1
            else:
                right = mid - 1
        return False
        """
```

#### 89.子集II（90-Meidum）

##### Key：回溯法-求包含重复元素数组的子集

题意：给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。解集 不能 包含重复的子集。返回的解集中，子集可以按任意顺序排列。

分析：这个题是78-子集的升级版，78题主要题意为求不包含重复元素数组的子集，开始写78题的时候刚开始写leetcode，还没有接触过回溯法所以选取的是一个比较容易理解的方法，遍历数组和结果集，在结果集中的每个子列表依次加入当前遍历的数，然后将新列表加入结果集中。本题所求为包含重复元素数组的子集，因为数组中包含重复元素结果集中不能包含重复子集，列出两种方法。

（1）如果在78题基础上要做两点改动，首先去重先对原数组进行排序，再遍历res内子列表添加新元素之后，判断一下是否已经在结果集中出现，没有出现的话就加入结果集中

（2）回溯法：判断当前路径是否在结果集中，与遍历未知区域时判断当前元素是否与前一元素重复如果重复则不回溯，这两个二选一即可达到本题的效果，注意去重之前先排序，重复元素才会相邻。

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
      # 从前往后遍历, 遇到一个数, 之前的所有集合添加上这个数, 组成新的子集，注意判断是否有重复子集
        res = [[]]
        # 去重先将原数组排序
        nums.sort()
        # 遍历数组中的数
        for i in range(len(nums)):
            size = len(res)
            # 遍历结果集内的子集
            for j in range(0,size):
                temp = list(res[j])
                # 将当前遍历到的数加入当前子集中
                temp.append(nums[i])
                # 判断当前结果集是否已经包含该子集，要求解集不能包含重复的子集
                if temp not in res:
                    res.append(temp)
        return res

        # 回溯法：求包含重复元素数组的子集
        """
        def subsetsWithDup(self, nums):
            res = []
            # 去重需要先排序，便于重复元素前后比较
            nums.sort()
            self.dfs(nums, 0, res, [])
            return res
        
        def dfs(self, nums, index, res, path):
            # 这里要是判断path是否在res中出现，下面就不用判断两个相邻元素是否相同
            # 这里要是直接append不判断path是否出现，下面在path加入元素的时候就要判断相邻元素是否重复再决定是否加入
            if path not in res:
                res.append(path)
            for i in range(index, len(nums)):
                # 依次判断nums元素是否加入路径当中，当前元素与上一个元素重复则不加入其中
                # if i > index and nums[i] == nums[i-1]:
                #     continue
                self.dfs(nums, i + 1, res, path + [nums[i]])
        """
```

#### 90.移动零（283-Easy）

##### Key：双指针

题意：给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

分析：使用双指针，左指针指向当前已经处理好的序列的尾部，右指针指向待处理序列的头部。右指针不断向右移动，每次右指针指向非零数，则将左右指针对应的数交换，同时左指针右移。注意到以下性质：左指针左边均为非零数；右指针左边直到左指针处均为零。因此每次交换，都是将左指针的零与右指针的非零数交换，且非零数的相对顺序并未改变。

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 双指针
        # 每次交换，都是将左指针的零与右指针的非零数交换，且非零数的相对顺序并未改变。
        left,right = 0,0
        while right < len(nums):
            # 利用右指针移动判断当前元素是否为0，非0的话与左指针元素进行交换，然后移动左右指针
            if nums[right] != 0:
                nums[left],nums[right] = nums[right],nums[left]
                left += 1
            right += 1
        
        # 只用一行代码搞定过的天秀，key = bool只对非零元素和零元素排序，非零元素不排序
        # 降序：将非零元素排在前面，0排在后面，不影响非零元素的相对位置
        # nums.sort(key=bool, reverse=True)
```

#### 91.笨阶乘（1006-Medium）

题意：通常，正整数 n 的阶乘是所有小于或等于 n 的正整数的乘积。例如，factorial(10) = 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1。相反，我们设计了一个笨阶乘 clumsy：在整数的递减序列中，我们以一个固定顺序的操作符序列来依次替换原有的乘法操作符：乘法(*)，除法(/)，加法(+)和减法(-)。例如，clumsy(10) = 10 * 9 / 8 + 7 - 6 * 5 / 4 + 3 - 2 * 1。然而，这些运算仍然使用通常的算术运算顺序：我们在任何加、减步骤之前执行所有的乘法和除法步骤，并且按从左到右处理乘法和除法步骤。另外，我们使用的除法是地板除法（floor division），所以 10 * 9 / 8 等于 11。这保证结果是一个整数。实现上面定义的笨函数：给定一个整数 N，它返回 N 的笨阶乘。

分析：求解没有括号的表达式，先乘除后加减，表达式运算需要用到栈这个数据结构，所以遇到乘除立即算，遇到加减先入栈。本题的运算符顺序是以[*, / , +, -]依次循环的，首先初始化一个栈，栈内放当前N元素，遍历[N-1,1]，循环运算符，遇到乘除弹出栈顶元素和当前元素进行操作，遇到加把当前数字入栈，遇到减把当前数字取反入栈，最后栈内求和即为结果。注意python的整数除法问题，如果存在负数的话，整数除法//低板除是向下取整不是向零取整，一般情况要求向零取整的，所以对 Python 的整数除法问题，可以用 `int(num1 / float(num2))` 来做，即先用浮点数除法，然后取整，或者使用库函数 `operator.truediv(num1, num2)`。

```python
class Solution:
    def clumsy(self, N: int) -> int:
        # 遇到乘除先计算，遇到加减先入栈
        op = 0
        stack = [N]
        # 倒序遍历[N-1, 1]，运算符按顺序循环[*, / , +, -]
        for i in range(N-1, 0, -1):
            if op == 0:
                stack.append(stack.pop() * i)
            elif op == 1:
                # 注意python中的整数除法
                stack.append(int(stack.pop()/float(i)))
            elif op == 2:
                stack.append(i)
            elif op == 3:
                stack.append(-i)
            op = (op + 1) % 4
        return sum(stack)
```

#### 92.直方图的水量（面试题17.21-Hard）

##### Key：双指针

题意：给定一个直方图(也称柱状图)，假设有人从上面源源不断地倒水，最后直方图能存多少水量?直方图的宽度为 1。上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的直方图，在这种情况下，可以接 6 个单位的水（蓝色部分表示水）。

分析：首先这道题的最佳解法应该是双指针，看到一个非常容易理解的思路就是利用双指针按行遍历，核心思想：`总体积减去柱子体积就是水的容量`，利用左右指针的下标差值计算出每一层的体积（雨水和柱子一起），sum（height）即为柱子的体积。所以我们遍历每一行来计算每一层的体积，利用high记录层数，当左右指针指向的区域高度小于high时，左右指针都向中间移动直到指针指向区域大于等于high的值，若不小于high则指针不移动，计算当前层的体积累加到volumn中，最后volumn - sum(height)即为结果。

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # 第一种方法：双指针，按行遍历，总体积减去柱子体积就是水的容量
        # 利用左右指针的下标差值计算出每一层体积
        Sum = sum(height)         # 柱子体积
        size = len(height)        # 区域长度
        left, right = 0, size - 1 # 双指针
        volumn, high = 0, 1       # 总体积和高度初始化
        while left <= right:
            # 当左右指针指向的区域高度小于high时，左右指针都向中间移动
            while(left <= right and height[left] < high):
                left += 1
            while(left <= right and height[right] < high):
                right -= 1
            # 直到指针指向大于等于high的时候，计算该层的体积，累加到volumn
            volumn += right - left + 1
            # 层数累加
            high += 1
        return volumn - Sum

        # 第二种方法：附加一下比较容易想到的暴力解法【简单解法】
        """
        # 对于每个位置，向左右找最高的木板；当前位置能放的水量是：左右两边最高木板的最低高度 - 当前高度
        res = 0
        # 第 0 个位置和 最后一个位置不能蓄水，所以不用计算
        for i in range(1, len(height) - 1):
            # 求右边的最高柱子
            rHeight = max(height[i + 1:])
            # 求左边最高柱子
            lHeight = max(height[:i])
            # 左右两边最高柱子的最小值 - 当前柱子的高度
            h = min(rHeight,lHeight) = height[i]
            # 如果能蓄水
            if h > 0:
                res += h
        return res
        """
```

#### 93.最长公共子序列（1143-Meidum）

##### Key：二维动态规划

题意：给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。如果不存在公共子序列，返回 0 。一个字符串的子序列是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的公共子序列是这两个字符串所共同拥有的子序列。

分析：状态定义：`dp[i][j]` 表示 `text1[0:i-1]` 和 `text2[0:j-1]` 的最长公共子序列。没有设置为`text1[0:i]`是方便当i = 0或者j = 0时，`dp[i][j]`表示的为空字符串和另外一个字符串的匹配，可以将其初始化为0。状态转移方程分为两种情况，当两个子字符串当前遍历到的末尾字符相同时，最长公共子序列长度在前面的基础上加1，当不相同时，最长公共子序列长度取两个子字符串分别除去当前不等字符其余子串与对方的匹配。最后结果就是`dp[M][N]`，表示两个字符串的最长公共子序列。

转移方程：`dp[i][j] = dp[i-1][j-1] + 1`and `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`

```python
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        """
        :type text1: str
        :type text2: str
        :rtype: int
        """
        # dp[i][j] 表示 text1[0:i-1](包括 i - 1) 和 text2[0:j-1](包括 j - 1) 的最长公共子序列
        M, N = len(text1), len(text2)
        # 便于当 i = 0 或者 j = 0 的时候，dp[i][j]表示的为空字符串和另外一个字符串的匹配
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        # dp[i][j] = 0，无需再遍历
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                # 两个子字符串的最后一位相等，所以最长公共子序列又增加了 1
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    # 两个子字符串的最后一位不相等，此时目前最长公共子序列长度为两个字符串其中之一去除当前不相等字符前面子串与另一字符串的匹配
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[M][N]
```

#### 94.森林中的兔子（781-Medium）（据说是3.31华为笔试题）

##### Key：取整函数

题意：森林中，每个兔子都有颜色。其中一些兔子（可能是全部）告诉你还有多少其他的兔子和自己有相同的颜色。我们将这些回答放在 answers 数组里。返回森林中兔子的最少数量。

分析：数组中出现的每一个数字key代表某一种颜色的兔子有(key+1)只，报相同数字的兔子有可能为一种颜色类，但是如果有多只兔子报同一个数字，并且报同一个数字的兔子的总数大于报的这个数字 + 1，那么肯定应该是颜色种类要多于一类。`math.ceil(value / (key + 1))`, 记录报相同数字的兔子最少种类数, value为报相同数字key的兔子的个数。如果刚好除尽，则说明报相同数字的兔子的颜色有value // (key + 1)种，否则相同数字的兔子的颜色有value // (key + 1) + 1 种，多出来报相同数字的兔子数的要单独用一种颜色，即比value // (key + 1)多一种，所以用向上取整函数。

```python
class Solution:
    def numRabbits(self, answers: List[int]) -> int:
        # 记录报相同数字的兔子最少种类数, value为报相同数字key的兔子的个数
        res = 0
        count = collections.Counter(answers)
        for key,value in dict(count).items():
            # (key + 1) * 颜色种类数 math.ceil(value / (key + 1)) 即为报该数字的兔子的最少总数
            # ceil()向上取整函数，也可以表示成 (n + x) // (x + 1)，//地板除向下取整取比目标结果小的最大整数
            res += math.ceil(value / (key + 1)) * (key + 1)
        return res
```

#### 95.删除有序数组中的重复项II(80-Meidum)

##### Key：双指针

题意：给你一个有序数组 nums ，请你原地删除重复出现的元素，使每个元素最多出现两次 ，返回删除后数组的新长度。不要使用额外的数组空间，你必须在原地修改输入数组 并在使用 O(1) 额外空间的条件下完成。

分析：本题意思为在一个有序数组中让每个数字最多出现两次，返回一个n，使得数组的前n个元素就是结果，在原数组上修改。需要设置两个指针，一个指向当前即将放置元素的位置，另一个向后遍历所有元素，`慢指针slow`指向当前即将放置元素的位置，`快指针fast`向后遍历所有元素，比较nums[fast]和nums[slow-2]，判断fast指向位置元素是否需要保留。

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        slow = 0
        for fast in range(len(nums)):
            # 设置快慢指针，慢指针指向待处理元素，快指针向前遍历寻找尚未重复元素
            if slow < 2 or nums[slow-2] != nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

#### 96.合并两个有序数组（88-Easy）

题意：给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。

分析：由于num1的空间有m+n，所以从后向前遍历两个数组，将nums2中与num1末尾元素比较，较大者放入num1的后面，然后继续向前遍历，设置k初始指向num1的最后位置每次将比较大的放入k位置，k在每次循环中均向前移动一个位置。

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # 从后向前比较，将nums2的数据比num1末尾数据大的放入nums1的后面
        k = m + n - 1
        while m > 0 and n > 0:
            if nums1[m-1] < nums2[n-1]:
                nums1[k] = nums2[n-1]
                n -= 1
            else:
                nums1[k] = nums1[m-1]
                m -= 1
            k -= 1
        # 遍历完成后 m 和 n 至少有一个为 0，
        # m == 0,将nums2元素直接复制到nums1的头部，n == 0,num1前面留在原地
        nums1[:n] = nums2[:n]
```

#### 97.删除有序数组中的重复项（26-Easy）

##### Key：双指针

题意：给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

分析：这个和LeetCode80思路相同，一个指向当前即将放置元素的位置，另一个向后遍历所有元素，`慢指针slow`指向当前即将放置元素的位置，`快指针fast`向后遍历所有元素，比较nums[fast]和nums[slow-1]（因为每个元素只出现一次），判断fast指向位置元素是否需要保留。

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # 设置快慢指针，慢指针指向的是当前应该插入的位置，快指针是要指向待插入元素的位置。
        slow = 0
        for fast in range(0, len(nums)):
            # 比较fast和slow-1，如果相等说明出现重复元素，fast向前跳过重复项找到不同元素进行覆盖
            if slow < 1 or nums[fast] != nums[slow-1]:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

#### 98.搜索旋转排序数组（33-Easy）

##### Key：旋转无重复元素的排序数组二分查找

题意：整数数组 nums 按升序排列，数组中的值 互不相同。在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如[0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1。

分析：搜索旋转无重复元素的排序数组，运用二分查找，判断mid和旋转点pivot的位置，给出的数组是旋转有序的，先对整体二分查找一次然后判断 mid 与 pivot 的前后位置区分两个部分有序的序列，对于部分有序继续使用二分查找寻找target。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 起始对整体做一次二分，判断mid与旋转点的位置对两个部分有序继续做二分
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            # 判断mid在旋转点之后还是之前，分别判断target
            if nums[mid] <= nums[right]:
                # mid指向旋转点之后，则 mid后面是有序的
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                # mid指向旋转点之前，则 mid前面是有序的
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
        return -1
```

#### 99.搜索旋转排序数组II（81-Meidum）

##### Key：旋转有重复元素的排序数组二分查找

题意：已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。

分析：搜索旋转有重复元素的排序数组，这个题和上面一道题思想差不多，代码也差不多，唯一区别就是数组有重复元素，这样在判断mid的时候有些问题，比如 [2,1,2,2,2] 和 [2,2,2,1,2] ，最开始时，ft、mid、right三者相等都为 2。如果我们想找 1，而这个 1 可以在 mid 的左边也可以在 mid 的右边。所以就不知道该在哪个区间继续搜索。一个解决本题的办法是，遇到 nums[left] == nums[right] 的情况时，直接向右移动 left，直至 nums[left] != nums[right]。转换为在 [1,2,2,2] 和 [2,2,2,1] 上搜索，1 所在的区间就可以根据 mid 和 right 的大小关系而获得。本题代码在上一题基础上增加left和right是否相等的判断，如果相等的话移动left并continue。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        # 起始对整体做一次二分，判断mid与旋转点的位置对两个部分有序继续做二分
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True
            # left和right均为重复元素时，无法判断在哪个区间继续搜索，直接向右移动 left 直到他们不相等
            if nums[left] == nums[right]:
                left += 1
                continue
            # 判断mid在旋转点之后还是之前，分别在两个有序序列中判断target位置
            if nums[mid] <= nums[right]:
                # mid指向旋转点之后，则 mid后面是有序的，继续做一次二分
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                # mid指向旋转点之前，则 mid前面是有序的，继续做一次二分
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
        return False
```

#### 100.寻找旋转排序数组中的最小值（153-Meidum）

##### Key：旋转无重复元素的排序数组最小值二分查找

题意：已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]。注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

分析：旋转排序数组的题目均可以使用二分查找，旋转点两边的子数组为单调递增的，判断mid与left和right的关系来判断最小值出现的位置。当左端点小于右端点时说明数组本身有序最小值直接为最左边，当左端点大于右端点时说明为旋转数组，判断mid位置来决定左右指针的移动，当只有两个元素的时候，由于left > right，所以最小值一定是right。

````python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        left, right = 0, len(nums) - 1
        # 如果left位置小于right，则说明数组有序直接返回left位置
        mid = left
        # 根据二分的中点mid的元素大小和nums[left]比较，判断mid在在旋转点之后还是之前
        # 输入的是旋转有序数组
        while nums[left] >= nums[right]:
            # 只有两个元素时，在这left一定比right大，所以最小就是right
            if left + 1 == right:
                mid = right
                break
            mid = (left + right) // 2
            # mid在旋转点左边，所以最小值在mid点右边，移动left指针
            if nums[mid] >= nums[left]:
                left = mid
            # mid在旋转点右边，所以最小值在mid点左边，移动right指针
            elif nums[mid] <= nums[right]:
                right = mid
        # mid位置元素即为最小值
        return nums[mid]
````

#### 101.寻找旋转排序数组中的最小值II（154-Hard）

##### Key：旋转有重复元素的排序数组最小值的二分查找

题意：已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,4,4,5,6,7] 在变化后可能得到：若旋转 4 次，则可以得到 [4,5,6,7,0,1,4]；若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]；注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。给你一个可能存在 重复 元素值的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

分析：本题属于是存在重复元素的旋转排序数组，关键在于重复元素的处理。

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # 二分法
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            # mid一定在右边，移动左指针
            if nums[mid] > nums[right]: 
                left = mid + 1
            # mid一定在左边，移动右指针
            elif nums[mid] < nums[right]: right = mid
            # 当 mid和 right相等的时候，遇到重复元素移动 right指针
            else: right = right - 1
        return nums[left]
```

#### 102.丑数（263-Easy）

题意：给你一个整数 n ，请你判断 n 是否为 丑数 。如果是，返回 true ；否则，返回 false 。丑数 就是只包含质因数 2、3 和/或 5 的正整数。

分析：将质因数统计到一个数组中，遍历数组中的质因数取模再反复除，最后判断该数是否为1，主要是依据题意来分析。

```python
class Solution:
    def isUgly(self, n: int) -> bool:
        # 负数和零一定不是丑数
        if n <= 0:
            return False
        factors = [2,3,5]
        # 对质因数依次取模再反复除，直到不再包含质因数2,3,5
        for factor in factors:
            while n % factor == 0:
                n //= factor
        # 判断剩下数字是否为1，为1是丑数，否则不是
        return n == 1
```

#### 103.丑数II（264-Medium）

题意：给你一个整数 `n` ，请你找出并返回第 `n` 个 **丑数**。**丑数** 就是只包含质因数 `2`、`3` 和/或 `5` 的正整数

分析：要生成第n个丑数，必须从第一个丑数1开始，向后逐渐寻找，丑数只包含2,3,5三个因子，所以生成方式就是在已经生成的丑数集合中乘以[2,3,5]而得到新的丑数。用还没乘过 2 的最小丑数乘以 2；用还没乘过 3 的最小丑数乘以 3；用还没乘过 5 的最小丑数乘以 5。然后在得到的数字中取最小，就是新的丑数。`dp[i]`表示第 i - 1 个丑数，最后返回`dp[i-1]`

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        # 生成方式：在已经生成的丑数集合中乘以 [2, 3, 5] 而得到新的丑数
        if n < 0 :
            return 0
        # dp[i] 表示第 i - 1 个丑数（注dp[0]表示第一个丑数）
        dp = [1] * n
        # 三个指针表示 pi 的含义是有资格同 i相乘的最小丑数的位置
        index2, index3, index5 = 0, 0, 0
        for i in range(1, n):
            # 每次我们都分别比较有资格同2，3，5相乘的最小丑数，选择最小的那个作为下一个丑数
            dp[i] = min(2 * dp[index2], 3 * dp[index3], 5 * dp[index5])
            # 判断dp是由哪个相乘得到的，它失去了同 i 相乘的资格，把对应的指针相加
            if dp[i] == 2 * dp[index2]: index2 += 1
            if dp[i] == 3 * dp[index3]: index3 += 1
            if dp[i] == 5 * dp[index5]: index5 += 1
        return dp[n - 1]
```

#### 104.最大数（179-Medium）

题意：给定一组非负整数 `nums`，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

**注意：**输出结果可能非常大，所以你需要返回一个字符串而不是整数。

分析：依次两两比较nums中的数字交换顺序拼接组合，保留拼接结果大的顺序。把数字转成字符串然后进行组合比较大小决定是否移动顺序位置，将nums中顺序确定好之后依次以字符形式拼接成结果字符串，注意特殊情况0的判断，如果nums列表为多个0则res字符串也会有多个0，实际答案只要一个0，所以要进行一下判断，最终结果字符串的首字符是否为0，如果是0的话res='0'。

```python
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        # nums列表比较两个数不同的拼接顺序的结果，进而决定它们在结果中的排列顺序
        res = ''
        # 相邻两个数字转变成字符串前后拼接比较大小，按照从前向后拼接结果较大排序
        for i in range(len(nums)-1):
            for j in range(i + 1, len(nums)):
                if str(nums[i]) + str(nums[j]) < str(nums[j]) + str(nums[i]):
                    nums[i],nums[j] = nums[j],nums[i]
        # 按照要求已经排好序的nums需要将里面元素依次转变字符串然后连接起来
        for x in nums:
            res += str(x)
        # 特殊情况判定，如果nums列表是多个0，转成字符串也是多个‘0’，但是实际要求只要一个‘0’
        # 所以判断最终结果字符串的首字符是否为0，因为如果nums中有不为0的字符一定排在0的前面
        if res[0] == '0':
            res = '0'
        return res
```

#### 105.二叉搜索树节点的最小距离（783-Easy）

##### Key：中序遍历、二叉搜索树BST

题意：给你一个二叉搜索树的根节点 `root` ，返回 **树中任意两不同节点值之间的最小差值** 。此题与[530. 二叉搜索树的最小绝对差](https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/) 相同

分析：二叉搜索树的中序遍历是有序的，这是解决所有二叉搜索树问题的关键。题目要求两个不同节点之间的最小差值，相当于求BST中序遍历得到的有序序列中所有相邻节点之间的最小差值，可以有两种方法。首先利用数组保存中序遍历结果，先遍历结果放在数组中，然后对数组中相邻元素求差得到所有差值的最小值。第二种方法可以记录中序遍历时候的上一个被访问的结点，利用`root.val - pre.val`求差值，对该值取最小就是题目要求。注意细节中序遍历第一个结点没有pre节点，所以初始化None，在求差值之前判断pre是否存在，如果不存在说明是首节点直接跳过求差值部分。

附加：二叉树的三种遍历方法

```python
# 先序遍历
def dfs(root):
    if not root:
        return
    执行操作
    dfs(root.left)
    dfs(root.right)
# 中序遍历
def dfs(root):
    if not root:
        return
    dfs(root.left)
    执行操作
    dfs(root.right)
# 后序遍历
def dfs(root):
    if not root:
        return
    dfs(root.left)
    dfs(root.right)
	执行操作
```

```python
class Solution(object):
    def minDiffInBST(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 初始化中序遍历时前序结点
        self.pre = None
        # 保存差值，初始化结点最大值
        self.minDiff = 10e6
        # 调用中序遍历函数
        self.inorderTraversal(root)
        return self.minDiff

    def inorderTraversal(self,root):
        if not root:
            return
        # 中序遍历先访问左子树
        self.inorderTraversal(root.left)
        # 判断前序结点是否存在，如果不存在则不计算差值，说明当前遍历的是中序遍历第一个结点
        if self.pre:
            self.minDiff = min(self.minDiff, root.val - self.pre.val)
        # 将pre设置成新的前序结点
        self.pre = root
        # 遍历右子树
        self.inorderTraversal(root.right)
```

#### 106.实现Trie（前缀树）

题意：Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 Trie 类：

Trie() 初始化前缀树对象。
void insert(String word) 向前缀树中插入字符串 word 。
boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。

分析：（只保存小写字符的）「前缀树」是一种特殊的多叉树，它的 TrieNode 中 chidren 是一个大小为 26 的一维数组，分别对应了26个英文字符 'a' ~ 'z'，也就是说形成了一棵 26叉树。前缀树的结构可以定义为这样，里面存储了两个信息：isWord 表示从根节点到当前节点为止，该路径是否形成了一个有效的字符串。children 是该节点的所有子节点。详情看链接好啦
链接：https://leetcode-cn.com/problems/implement-trie-prefix-tree/solution/fu-xue-ming-zhu-cong-er-cha-shu-shuo-qi-628gs/

```python
class Node(object):
    def __init__(self):
        # 26个英文字符，{字符：Node}
        self.children = collections.defaultdict(Node)
        # isWord 表示从根节点到当前节点为止，该路径是否形成了一个有效的字符串
        self.isword = False

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node()


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        current = self.root
        # 拆解关键词，每个字符按照其在 'a' ~ 'z' 的序号，放在对应的 chidren 里面。下一个字符是当前字符的子节点
        for w in word:
            current = current.children[w]
        # 从根结点到当前结点标记为有效路径
        current.isword = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        current = self.root
        # 依次遍历该关键词所有字符，在前缀树中找出这条路径。
        for w in word:
            current = current.children.get(w)
            # 到某个位置路径断了则不存在
            if current == None:
                return False
        # 找到了这条路径，取决于最后一个结点的isWord标志
        return current.isword

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        current = self.root
        # 遍历当前字符
        for w in prefix:
            # 依次查询存在该前缀字符
            current = current.children.get(w)
            # 找不到说明不存在
            if current == None:
                return False
        # 均遍历完找到则返回True
        return True
```

#### 107.打家劫舍II(213-Medium)

##### Key:动态规划、环形数组处理方式

题意：你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，能够偷窃到的最高金额。

分析：打家劫舍I的升级版，增加条件是两个相邻的房子不能同时偷，首尾两房子相邻。`dp[i]`为**数组的前 i 个元素**中按照「两个相邻的房间不能同时偷」的方法，能够获取到的最大值。状态转移方程：dp[i] 有两种抉择：nums[i] 选或者不选，初始化为0，dp[0] = nums[0], dp[1] = max(nums[0], nums[1]),` dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])`。当nums[i]选择的时候，需要在前两个位置加上当前金额，不选的时候即为i-1时的金额，结果即为dp[N-1]，环形数组的处理方式是分割为两个队列[0:N-1]和[1,N]，比较两种情况的最大值

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        N = len(nums)
        if N == 0:
            return 0
        if N == 1:
            return nums[0]
        # 将环拆成两个队列，[0, N-1]和[1, N]
        return max(self.rob1(nums[0:N-1]), self.rob1(nums[1:N]))
    # 非环数组情况，打家劫舍I
    def rob1(self, nums: List[int]) -> int:
        N = len(nums)
        if N == 0:
            return 0
        if N == 1:
            return nums[0]
        # dp[i]表示满足不触动警报条件前i家可以偷窃到的最高金额
        dp = [0] * N
        # 只有一家时只能选择该家
        dp[0] = nums[0]
        # 有两家时要选一家金额最多的
        dp[1] = max(nums[0],nums[1])
        for i in range(2, N):
            # 是否选择当前i位置家进行打劫，如果选择则在i-2基础上加，如果不选即为dp[i-1]
            dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        return dp[N-1]
```

#### 108.打家劫舍(198-Medium)

##### Key:动态规划

题意：你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

分析：就是上一题的rob1，注意状态定义和状态转移方程

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        N = len(nums)
        if N == 0:
            return 0
        if N == 1:
            return nums[0]
        # 状态定义：dp[i]为满足不触动警报条件下前i家偷窃到的累计最高金额
        dp = [0] * N
        dp[0] = nums[0]
        dp[1] = max(nums[0],nums[1])
        # 当前是否选择nums[i]，比较选择或者不选的最值定义状态转移方程
        for i in range(2, N):
            dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        return dp[-1]

```

#### 109.打家劫舍III（337-Medium）

##### Key: 树形dp

题意：在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

分析：第三版在于其结构从线性结构转化为树形结构，可采用动态规划递归处理，对于每个结点定义偷和不偷两种状态，分别计算其最大收益。定义方法以当前传入节点为根节点计算抢劫该节点和不抢劫该节点分别获得的最大收益将其返回，将左右子树传入当前方法递归，最后比较选取最大值。

```python
class Solution:
    def rob(self, root: TreeNode) -> int:
        # _rob方法表示以root为根节点的树中，返回抢劫根节点与不抢劫根节点可获得的最大值
        def _rob(root):
             # 两个0分别表示偷/不偷该节点可获得的最大值
            if not root: return 0, 0 
            # left和right均为二维数组，第一维表示偷该节点获取的金额，第二维表示不偷该节点获取的金额
            # 递归对于以root.left为根节点的树计算抢劫根节点和不抢劫根节点可获得的最大金额，right同理
            left = _rob(root.left)
            right = _rob(root.right)
            # 偷当前节点, 则左右子树都不能偷
            v1 = root.val + left[1] + right[1]
            # 不偷当前节点, 则取左右子树中最大的值
            v2 = max(left) + max(right)
            return v1, v2

        return max(_rob(root))
```

#### 110.最长有效括号(32-Hard)

题意：给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

分析：用栈解决从前往后遍历字符串。栈条件为1.栈为空 2.当前字符是'(' 3.栈顶符号位')'，三种条件都没办法消去成对的括号。符合消去成对括号时，拿当前下标减去栈顶下标。

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = []
        res = 0
        N = len(s)
        for i in range(N):
            # 将元素下标入栈：栈为空或当前元素为左括号或栈顶元素为右括号三种情况都不能进行配对消除
            if not stack or s[i] == '(' or s[stack[-1]] == ')':
                stack.append(i)
            else:
                # 表明当前有效可以匹配
                stack.pop()
                # 这里为什么先弹出后减是要匹配最远的也就是最先出栈的，它当初的位置在当前栈顶位置之后
                # 样例：")()())"，如果先计算长度的话结果就是2，因为计算的是第二个完整括号的长度，如果先弹出后计算的话，第二个完整右括号减去第一个当前栈顶右括号即为当前右括号到第一个完整括号的左括号匹配长度
                # 我说的有点啰嗦我懂就好不我以后一定不懂
                res = max(res, i - (stack[-1] if stack else -1))
        return res
```

#### 112.前K个高频元素（347-Meidum）

##### Key：堆排序

题意：给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

分析：首先先对数组内元素以及出现次数进行统计，然后排序找出前k个最大的输出它们的key，开始使用库函数，后来感觉如果笔试不让调用库函数怎么办，但是后来发现如果采用堆排序的话还是不可避免调用函数和模块。讲一下堆排序吧，首先还是先需要将数组内元素和对应出现次数统计出来，相当于接下来是寻找前K个最大的值的问题，构造一个最小堆，当堆内元素不足k个时候将(val,key)入堆【替换顺序是让堆根据出现次数val排序】，当堆内元素大于k个时候判断当前值和堆顶元素值进行比较，如果比堆顶元素大则替换堆顶元素，保证堆内元素为前K个最大。

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 记录元素出现次数
        count = collections.Counter(nums)
        # 构造一个小顶堆：放入元素和出现次数
        heap = []
        for key,val in count.items():
            # 当堆内元素大于等于k个时判断是否需要替换
            if len(heap) >= k:
                # 当前元素值大于堆顶元素，进行替换
                if val > heap[0][0]:
                    heapq.heapreplace(heap, (val,key))
            # 堆内元素个数小于k继续push
            else:
                heapq.heappush(heap, (val,key))
        # 最后堆内元素即为次数出现前k的值键对，遍历取值
        return [item[1] for item in heap]
```

#### 113.最小路径和(64-Meidum)

##### Key:动态规划

题意：给定一个包含非负整数的 `m x n` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。**说明：**每次只能向下或者向右移动一步。

分析：注意只能向下或者向右走，`dp[i]`定义为从左上角到(i,j)位置的最短路径和，左上角元素单独初始化定义，对于第一行和第一列只能有单方向的来源所以单独判断加上当前元素，其他位置利用状态转移方程寻找两个方向哪个路径短就取哪个加上当前长度`dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i][j]`。

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        # 空判
        if not grid or not grid[0]:
            return 0
        
        # dp[i][j]表示从左上角到(i,j)位置的最短路径和
        M, N= len(grid), len(grid[0])
        dp = [[0] * N for _ in range(M)]

        # 左上角元素单独定义
        dp[0][0] = grid[0][0]

        # 对于第一行元素，路径只能来自于左边元素向右走加上当前位置
        for j in range(1, N):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        # 对于第一列元素，路径只能来自于上边元素向下走加上当前位置
        for i in range(1, M):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        # 对于其他元素，路径来自于上面或者左边路径的最小值加上当前元素
        for i in range(1, M):
            for j in range(1, N):
                dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i][j]
        return dp[M-1][N-1]
```

#### 114.扰乱字符串（87-Hard）

题意：题目内容太长所以放链接上来https://leetcode-cn.com/problems/scramble-string/

分析：分析也直接放链接了好吧这题很难理解我感觉https://leetcode-cn.com/problems/scramble-string/solution/fu-xue-ming-zhu-ji-yi-hua-di-gui-by-fuxu-r98z/

```python
class Solution:
    # 避免递归时出现的重复计算
    @functools.lru_cache(None)
    def isScramble(self, s1: str, s2: str) -> bool:
        # 从一个位置将两个字符串分别划分成两个子串，递归判断两个字符串的两个子串是否互相为「扰乱字符串」
        N = len(s1)
        if N == 0:return True
        if N == 1:return s1 == s2
        if sorted(s1) != sorted(s2):
            return False
        # 两种情况遍历每个位置进行分割
        # 在判断是否两个子串能否通过翻转变成相等的时候，需要保证传给函数的子串长度是相同的
        # 左子树的两个字符串的长度都是 i, 右子树的两个字符串的长度都是 N - i。
        # 如果上面两种情况有一种能够成立，则 s1 和 s2 是「扰乱字符串」。
        for i in range(1, N):
            if self.isScramble(s1[:i],s2[:i]) and self.isScramble(s1[i:],s2[i:]):
                return True
            elif self.isScramble(s1[:i],s2[-i:]) and self.isScramble(s1[i:],s2[:-i]):
                return True
        return False
```

#### 116.存在重复元素（220-Meidum）

##### Key：桶排序

题意：给你一个整数数组 nums 和两个整数 k 和 t 。请你判断是否存在 两个不同下标 i 和 j，使得 abs(nums[i] - nums[j]) <= t ，同时又满足 abs(i - j) <= k 。如果存在则返回 true，不存在返回 false。

分析：学习参考视频：https://www.bilibili.com/video/BV1Ty4y147KD?from=search&seid=17746428523113060281，因为从来没有涉及到桶排序所以第一次学习，本题设计的条件绝对值之差刚好符合桶排序的一些性质。首先定义桶的大小，将数组内所有元素依据公式求出对应桶的id，利用哈希表将桶id和对应元素存储起来，先确认一下该桶是否已经被创建过，如果已经被创建过说明内部一定有一个元素，和当前元素正好可以放在一个桶，同一桶内元素满足`abs(nums[i] - nums[j]) <= t`直接返回True，如果桶id不存在则创建加入字典，接着判断相邻桶内的对应元素是否满足`abs(nums[i] - nums[j]) <= t`，满足则返回True。对于k这个条件的判断，在遍历过程中i<k的时候一定满足，当走到i >= k的时候，还没有返回接下来i就即将+1了，当i变成i+1时i-k位置的桶已经没有意义了，所以我们提前将旧桶删除。

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        # 桶排序，地板除t+1遍历数组得到应放入桶的序号，利用哈希表存储桶和里面对应元素
        # 遍历元素得到桶id检查当前桶是否已被创建，如果被创建说明里面有一元素，当前元素和其满足条件返回True
        # 如果没有被创建则创建放入元素，加入字典关系
        # 桶内只有一元素时判断相邻桶内对应位置元素是否会出现绝对值小于等于t的情况
        # 当前遍历元素位置超过k的时候，就需要把i-k的数从桶里删除，因为不再考虑他们了
        all_buckets = {}    # 桶的集合
        bucket_size = t + 1
        for i in range(len(nums)):
            # i < k 的情况
            bucket_id = nums[i] // bucket_size
            # 桶已存在说明已有元素，当前元素也属于该桶，二者绝对值之差必然小于等于t
            if bucket_id in all_buckets:
                return True
            # 不存在即创建
            all_buckets[bucket_id] = nums[i]
            # 判断相邻桶内元素
            if (bucket_id - 1) in all_buckets and abs(all_buckets[bucket_id-1] - nums[i]) <= t:
                return True
            if (bucket_id + 1) in all_buckets and abs(all_buckets[bucket_id+1] - nums[i]) <= t:
                return True
            # i >= 没有返回True接下来i马上进入i+1，当i变为i+1的时候，i-k位置的桶已经不再考虑，所以删除
            if i >= k:
                all_buckets.pop(nums[i - k] // bucket_size)
        return False
            
```

#### 117.寻找重复数（287-Meidum）

题意：给定一个包含 n + 1 个整数的数组 nums ，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设 nums 只有一个重复的整数 ，找出这个重复的数 。

分析：利用哈希表记录出现的次数，在数组遍历过程中在哈希表中查找键，如果不存在的话get方法返回默认值0并将其值更新为1，判断如果当前键值为2的话说明是重复元素。

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # 定义一个哈希表
        hashtable = dict()
        for num in nums:
            # 查找数组中的键
            # 如果没有的话返回默认值0，将其值更新为1插入到字典当中，如果有的话则在其值1的基础上+1
            hashtable[num] = hashtable.get(num,0) + 1
            # 当某个键的值为2的时候，说明为重复值
            if hashtable[num] == 2:
                return num
```

#### 118.字母异位词分组

题意：给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

分析：字母异位词有两个性质，两个字符串如果互为字母异位词的话他们排序后的字符串相同，而且对应字母出现的次数相同。本题解法利用他们排序后的字符串相同，将其作为哈希表的key，将该key下的str加入值列表中，最后返回哈希表值列表。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 建立哈希表，映射每一组字母异位词，所以构造为列表形式的dict
        hashtable = collections.defaultdict(list)
        for str in strs:
            # 将字符串排序后得到的字符串作为键值，因为字母异位词排序后的字符串一定相同
            key = "".join(sorted(str))
            # 将对应字母异位词作为值加入到key对应的值列表中
            hashtable[key].append(str)
        # 将所有values以list形式返回
        return list(hashtable.values())
```

#### 119.移除元素（27-Easy）

题意：给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

分析：可以采用双指针法和通用思考的方法，双指针法设置左右指针，初始化的时候左指针指向数组开始，右指针指向数组末尾，在遍历过程中左指针指向val时用右指针所指内容覆盖，然后右指针右移，如果左指针没有指向待删除元素左指针继续向前。通用思考法遍历数组当前元素x如果与val相同，那么跳过该元素，如果不相同将其放到下标 idx 的位置，并让 idx 自增右移。

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        # 双指针在数组首尾，向中间遍历该序列
        left, right = 0, len(nums)-1
        while left <= right:
            # 左指针指向重复元素时，用序列右边右指针内容覆盖，然后右指针右移
            if nums[left] == val:
                nums[left] = nums[right]
                right -= 1
            # 左指针继续向前
            else:
                left += 1
        return left

        # 另一种方法
        # 如果当前元素 x 与移除元素 val 相同，那么跳过该元素。
        # idx = 0;
        # for x in nums:
        #     if (x != val):
        # 如果当前元素 x 与移除元素 val 不同，那么将其放到下标 idx 的位置，并让 idx 自增右移。
        #         nums[idx] = x;
        #         idx += 1
        # return idx;
```

#### 120.实现strStr()(28-Easy)

题意：实现 strStr() 函数。给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。

分析：最开始想到的就是滑动窗口，利用左右指针来遍历截取原字符串判断是否与子串相等，相等的话返回left初始位置，写了一下提交之后感觉时间和空间复杂度都还可以，时间89.52%，空间85.87%，看了看题解都是KMP算法感觉好复杂不是很好懂的样子。

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        # 双指针滑动窗口判断是否包含子串
        if not needle:
            return 0
        left,right = 0, len(needle)
        # 因为在字符串截取时左闭右开，所以right可以取到len(haystack)
        while right <= len(haystack):
            if haystack[left:right] == needle:
                return left
            left += 1
            right +=1
        return -1
```

#### 122.解码方法(91-Medium)

##### Key:动态规划

题意：一条包含字母 `A-Z` 的消息通过以下映射进行了 **编码**，A到Z映射成1-26，给你一个只含数字的 **非空** 字符串 `s` ，请计算并返回 **解码** 方法的 **总数** 。因为 `"06"` 不能映射为 `"F"` ，这是由于 `"6"` 和 `"06"` 在映射中并不等价。

分析：状态定义为题目`dp[i]表示前i个字符的解码方法数`，状态转移方程有两种情况，当只用当前一个字符编码时，`dp[i]`取决于`dp[i-1]`的解码方法数，当用前后两个字符编码时，`dp[i]`取决于`dp[i-2]`的解码方法数，i>1的位置均会有两种情况，将两种情况累加到`dp[i]`中，注意初始化时dp[0]=1作为基础，在第一个字符编码的时候只有一种方法来源于dp[0]，s中下标均减1来对应dp中1-N的位置。

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        # dp[i]表示前i个字符的解码方法数
        N = len(s)
        # 0位置作为基础，1-n对应s中0-n-1
        dp = [0] * (N+1)
        # 计算dp[1]的时候一个字符解码方式只有一种来源于dp[0]
        dp[0] = 1
        for i in range(1, N+1):
            # 使用一个字符s[i-1]进行解码，s[i-1]不为'0'时，dp[i]取决于dp[i-1]，加入dp[i-1]解码方法数
            if s[i-1] != '0':
                dp[i] += dp[i-1]
            # 使用了两个字符即s[i-2]和s[i−1]进行解码，要满足s[i-2]不为‘0’且两个字符对应的数字在26之内
            # dp[i]取决于dp[i-2]，加入dp[i-2]解码方法数
            if i > 1 and s[i-2] != '0' and int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]
        return dp[N]
```

#### 123.买卖股票的最佳时机(121-Easy)

##### Key:动态规划

题意：给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

分析：定义状态`dp[i]为前i天获得的最大收益`，初始化为0，前i天获得的最大收益：可以由前i-1天获得的最大收益与当前卖出可获得的最大收益相比较，当前卖出的最大收益由当天的价格减去前i-1天的最低价格得到，定义变量在遍历过程中记录遍历到的最低价格。状态转移方程为`dp[i] = max(dp[i-1], prices[i] - min_price)`

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        N = len(prices)
        # dp[i]表示前i天获得的最大收益
        dp = [0] * N
        # 记录目前遍历过的最小值
        min_price = prices[0]
        for i in range(1, N):
            # 前i天获得的最大收益等于：max(前i-1天获得的最大收益与当天卖出可获得最大收益)
            dp[i] = max(dp[i-1], prices[i] - min_price)
            # 记录前i-1天的最小值
            min_price = min(prices[i], min_price)
        return dp[N-1]
```

#### 124.矩阵区域不超过K的最大数值和（363-Hard）

题意：给你一个 m x n 的矩阵 matrix 和一个整数 k ，找出并返回矩阵内部矩形区域的不超过 k 的最大数值和。题目数据保证总会存在一个数值和不超过 k 的矩形区域。

分析：将二维转为一维，依次累计列至行列表中，累加过程中在行列表中判断矩形区域是否满足和记录最大数值和，lst那边是不是lst[i]表示在sums中i位置之前的累加和这个地方不是很清晰。

```python
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        # 记录行和列
        row , col = len(matrix) , len(matrix[0])
        # 记录矩阵和
        res = float("-inf")
        # left和right索引表示列的范围
        for left in range(col):
            # 列累加和列表
            sums = [0] * row
            for right in range(left,col):
                for j in range(row):
                    # 同一行的左右列逐渐相加，转换为在一维数组sums中判断最大矩形和
                    sums[j] += matrix[j][right]
                # 存放对于sums中元素累加的和，lst[i]表示在sums中i位置之前的累加和
                lst = [0]
                # 用来累加之前算出来的累加列表
                cur = 0
                for num in sums:
                    cur += num
                    # 寻找cur-k是否存在用来判断sums(0,j)-sums(0,i-1)<=k是否满足
                    loc = bisect.bisect_left(lst,cur-k)
                    if loc < len(lst):
                        # 存在记录当前区域的数值和
                        res = max(cur-lst[loc],res)
                    # 插入元素保持升序
                    bisect.insort(lst,cur)
        return res
```

#### 125.最大整除子集（368-Meidum）

##### Key：动态规划

题意：给你一个由 无重复 正整数组成的集合 nums ，请你找出并返回其中最大的整除子集 answer ，子集中每一元素对 (answer[i], answer[j]) 都应当满足：answer[i] % answer[j] == 0 ，或answer[j] % answer[i] == 0如果存在多个有效解子集，返回其中任何一个均可。

分析：动态规划设置dp数组为每个位置对应的最长整除子集，子集包含内元素均可以被nums[i]整除，定义状态为`dp[i]表示i位置对应元素的最长整除子集`，初始化每个位置只有其nums[i]自身元素，将数组排序后就可以将题目中两个或的条件转换成一个只需判断num[i]%nums[j]==0，遍历数组寻找每个位置前存在的因数，加入到dp[i]当中，当前dp[i]取决于dp[j]加入当前nums[i]与dp[i]二者中较大者，状态转移方程为`dp[i] = max(dp[j] + [nums[i]], dp[i], key=len)`，注意使用max函数需要制定key=len，按照长度进行选择。

```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        # dp[i]表示i位置对应元素的最长整除子集（里面所有元素均为nums[i]的因数）
        N = len(nums)
        if not nums: return nums
        if N == 1: return nums

        # 排序过后只存在nums[i] % nums[j] == 0的情况
        nums.sort()

        # 所有位置子集初始化为自身元素，自己是自己的因数
        dp = [[num] for num in nums]

        # 寻找i位置前所有nums[i]的因数加入dp[i]
        for i in range(1, N):
            # for j in range(i-1, -1, -1):
            for j in range(0, i):
                if nums[i] % nums[j] == 0:
                    # 如果nums[j]是nums[i]的因数，那么dp[j]全部均为nums[i]的因数
                    # 比较当前因数子集的长度与由dp[j]加入nums[i]后因数子集的长度，选取最长者
                    dp[i] = max(dp[j] + [nums[i]], dp[i], key=len)
                    
        # 最后返回所有位置中最大整除子集, 比较关键条件为长度
        return max(dp, key=len)
```

#### 126.颜色分类（75-Meidum）

题意：给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

分析：利用三个指针，左右指针分别指向数组的两端，i指针遍历数组遇到0和2的时候分别与左指针和右指针交换，注意此时i指针不移动，因为交换过来的数还未能确定位置，左右指针可以确定位置向中间移动。

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 三路排序：左右双指针，i遍历当前数组，
        N = len(nums)
        i = 0
        left = 0
        right = N - 1
        
        # i在left到right范围内时：
        # 为0放到左指针位置，为2放到右指针位置，这两种情况i指针不移动因为交换过来的数还未确定
        while i < N:
            if nums[i] == 0 and i > left:
                nums[i], nums[left] = nums[left], nums[i]
                left += 1
            elif nums[i] == 2 and i < right:
                nums[i], nums[right] = nums[right], nums[i]
                right -= 1
            else:
                i += 1
```

#### 128.递增顺序搜索树（897-Easy）

题意：给你一棵二叉搜索树，请你 **按中序遍历** 将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，并且每个节点没有左子节点，只有一个右子节点。

分析：把一棵「二叉搜索树」按照**中序遍历**构成一棵每个节点都只有右孩子的树，在中序遍历时，在访问根节点前，上一个被访问的节点是其左子树的最右下角的节点。只需要一个变量 `prev` 保存在中序遍历时，上一次被访问的节点。把当前节点 root.left 设置为 null；把 prev.right 设置为当前遍历的节点 root；
把当前 root 设置为 prev。这样的话，就保证了在中序遍历的过程中的访问顺序，形成了一个新的只有右孩子的树。

```python
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        # 新的树的根节点
        dummpy = TreeNode(-1)
        # 保存中序遍历上一个被访问的节点，上一个被访问的节点是其左子树的最右下角的节点
        self.prev = dummpy
        # 中序遍历
        self.inOrder(root)
        return dummpy.right
    
    def inOrder(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        # 中序遍历左子树
        self.inOrder(root.left)

        # 以下三步保证了在中序遍历的过程中的访问顺序，形成了一个新的只有右孩子的树
        # 将当前结点左结点设置为NULL
        root.left = None

        # 将前一个访问结点的右子树设置为当前访问结点
        self.prev.right = root
        # prev移动，root变成上一个被访问的结点
        self.prev = root

        # 中序遍历右子树
        self.inOrder(root.right)
```

#### 129.二叉搜索树(938-Easy)

题意：给定二叉搜索树的根结点 `root`，返回值位于范围 *`[low, high]`* 之间的所有结点的值的和。

分析：二叉搜索树最重要的性质：中序遍历是有序的，定义为每个节点的所有左子树都小于当前节点；每个节点的所有右子树都大于当前节点。直接使用题目给出的函数作为递归函数，重要的理解递归函数的含义：它的含义是寻找以 `root` 为根节点的所有 `[low, high]` 范围内的节点值之和，利用二叉搜索树的性质，左子树一定比root小，如果root.val <= low的话，那么就不用继续搜索左子树，二叉搜索树的右子树一定比root大，所以如果root.val >= high，那么不用继续搜索右子树。

```python
class Solution:
    def rangeSumBST(self, root, low, high):
        res = 0
        if not root:
            return res
        # 当前结点大于low的情况下继续搜寻左子树，如果小于low则无需查找左子树，因为左子树一定小于root.val
        if root.val > low:
            res += self.rangeSumBST(root.left, low, high)
        # 在范围内满足累加res
        if low <= root.val <= high:
            res += root.val
        # 当前结点小于high的情况下继续搜寻右子树，如果大于high则无需查找右子树，因为右子树一定大于root.val
        if root.val < high:
            res += self.rangeSumBST(root.right, low, high)
        return res
```

#### 130.平方数之和(633-Medium)

##### Key：双指针

题意：给定一个非负整数 `c` ，你要判断是否存在两个整数 `a` 和 `b`，使得 `a2 + b2 = c` 。

分析：双指针方法，i指针从0开始，j从可取的最大数`int(math.sqrt(c))`，移动过程中不断判断平方和是否等于c，如果小于c则移动左指针将结果变大，如果大于c则移动右指针将结果变小。

```python
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        # 双指针，注意j的最大值可以设置为c的平方根
        i = 0
        j = int(math.sqrt(c))
        # 利用双指针不断判断是否平方和等于c
        while i <= j:
            res = i * i + j * j
            if res == c:
                return True
            # 当前结果小于c移动左指针，增加res
            elif res < c:
                i += 1
            # 当前结果大于c移动右指针，减小res
            elif res > c:
                j -= 1
        return False
```

#### 131.青蛙过河(403-Hard)

##### Key：记忆化回溯

题意：一只青蛙想要过河。 假定河流被等分为若干个单元格，并且在每一个单元格内都有可能放有一块石子（也有可能没有）。 青蛙可以跳上石子，但是不可以跳入水中。给你石子的位置列表 stones（用单元格序号 升序 表示）， 请判定青蛙能否成功过河（即能否在最后一步跳至最后一块石子上）。开始时， 青蛙默认已站在第一块石子上，并可以假定它第一步只能跳跃一个单位（即只能从单元格 1 跳至单元格 2 ）。如果青蛙上一步跳跃了 k 个单位，那么它接下来的跳跃距离只能选择为 k - 1、k 或 k + 1 个单位。 另请注意，青蛙只能向前方（终点的方向）跳跃。

分析：每次有三种走法，要尝试每种可能肯定存在重复计算，所以加上备忘录@functools.lru_cache(None) ，状态定义和参数：`s(i, step)` 表示当前是第i块时候，通过step步过来的，决策和状态转移：接下来就是根据走的步数做状态变换，这里状态变换只有三种 step-1, step, step + 1, 尝试每种可能的走法，看能不能走到最后一块。

```python
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        if stones[1] - stones[0] > 1: return False

        stonesSet = set(stones) # 变成Set， 加速检索

        @functools.lru_cache(None) #加上备忘录，去掉重复计算
        def helper(i, step):
            # 状态，表示当前是第几块石头，是走几步走过来的。
            if i == stones[-1]:
                return True

            # 选择， 走 step + 1 步， 走 step 步，还是走step - 1 步？，
            # 只要往前走的步数有石头（在数组内），就试着可以往前走
            if i + step + 1 in stonesSet:
                if helper(i + step + 1, step + 1):
                    return True

            if i + step in stonesSet:
                if helper(i+ step, step):
                    return True
            
            if step - 1 > 0 and i + step - 1 in stonesSet:
                #这边要检查一下，step -1 要大于0 才走
                if helper(i+ step - 1, step -1):
                    return True

            return False

        return helper(stones[1], stones[1] - stones[0])
```

#### 132.只出现一次的数字II(137-Meidum)

题意：给你一个整数数组 `nums` ，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次 。**请你找出并返回那个只出现了一次的元素。

分析：为了稍微可以减少一点函数的依赖，所以自己利用字典计数，存储对应的数字和出现的次数，然后遍历字典寻找出现次数为1的键返回。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # 定义一个哈希表存储出现数字和对应次数
        hashTable = dict()
        
        for num in nums:
            # hashTable.get(num,0)如果没有该键的话将值设为0并返回默认值0
            # 出现num将其值更新加1插入到字典当中
            hashTable[num] = hashTable.get(num,0) + 1
        
        # 遍历字典寻找value值为1的key
        for key,value in hashTable.items():
            if value == 1:
                return key         
```

#### 134.只出现一次的数字III（260-Medium）

题意：给定一个整数数组 `nums`，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。你可以按 **任意顺序** 返回答案。

分析：让我自己写我还是想用哈希表，最容易理解而且也容易扩展，统计出现次数，再遍历出现次数为1的加入结果列表。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        # 定义哈希表
        hashmap = dict()
        res= []
        # 存储出现数字及其对应出现次数
        for i in nums:
            hashmap[i] = hashmap.get(i, 0) + 1
        # 将出现次数为1的键加入结果列表
        for i in hashmap:
            if hashmap[i] == 1:
                res.append(i)
        return res
```

#### 135.员工的重要性(690-Easy)

##### key:深度优先遍历dfs

题意：给定一个保存员工信息的数据结构，它包含了员工 唯一的 id ，重要度 和 直系下属的 id 。

比如，员工 1 是员工 2 的领导，员工 2 是员工 3 的领导。他们相应的重要度为 15 , 10 , 5 。那么员工 1 的数据结构是 [1, 15, [2]] ，员工 2的 数据结构是 [2, 10, [3]] ，员工 3 的数据结构是 [3, 5, []] 。注意虽然员工 3 也是员工 1 的一个下属，但是由于 并不是直系 下属，因此没有体现在员工 1 的数据结构中。

现在输入一个公司的所有员工信息，以及单个员工 id ，返回这个员工和他所有下属的重要度之和。

分析：先利用哈希表将员工ID和员工信息存储起来便于利用id查找，构建dfs函数返回指定id重要度与其所有下属重要度之和累加，下属重要度利用dfs递归求取。

```python
class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        # 先用哈希表存储员工ID及其员工信息[id:e]
        hashTable = dict()
        for e in employees:
            hashTable[e.id] = e
        # 深度优先遍历，对指定ID员工返回其重要度加遍历下属的重要度之和
        def dfs(ID) -> int:
            e = hashTable[ID]
            return e.importance + sum(dfs(subID) for subID in e.subordinates)
        # 初始对指定id进行dfs
        return dfs(id)
```

#### 136.解码异或后的数组(1720-Easy)

##### Key：异或性质

题意：未知 整数数组 arr 由 n 个非负整数组成。经编码后变为长度为 n - 1 的另一个整数数组 encoded ，其中 encoded[i] = arr[i] XOR arr[i + 1] 。例如，arr = [1,0,2,1] 经编码后得到 encoded = [1,2,3] 。给你编码后的数组 encoded 和原数组 arr 的第一个元素 first（arr[0]）。请解码返回原数组 arr 。可以证明答案存在并且是唯一的。

分析：异或运算的特点：(1)一个值与自身的异或总是为0：`x ^ x = 0`（2）一个值与0异或等于本身：`x ^ 0 = x`（3）可交换性：`a ^ b = b ^ a`（4）可结合性：`(a ^ b) ^ c = a ^ (b ^ c)`，根据以上的性质可以推导出由`a ^ b = c`可以的到`a = c ^ b`，由原数组第一个元素和编码数组可以依次解码出原数组的元素，然后利用上一个解码的再去解码下一个。

```python
class Solution:
    def decode(self, encoded: List[int], first: int) -> List[int]:
        # 异或满足交换律
        #  c^b = a
        #  c^a = b
        res = [first]
        for x in encoded:
            # 先利用first解码出原数组第二个元素，再利用第二个异或encoded求解第三个，以此类推
            # res[-1]表示上一个解码的元素
            res.append(x^res[-1])
        return res
```

#### 137.二叉树展开为链表(114-Meidum)

题意：给你二叉树的根结点 root ，请你将它展开为一个单链表：展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。展开后的单链表应该与二叉树 先序遍历 顺序相同。

分析：其实是将二叉树展开一个只有右子树的单边树按照先序遍历的顺序，先递归展开其左右子树，然后将展开的左右子树依次接入根节点即可。

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return root
        # 递归将左右子树展开
        self.flatten(root.left)
        self.flatten(root.right)
        
        # 临时保存已经展开的右子树
        temp = root.right
        # 将已经展开的左子树接入根节点的右边
        root.right = root.left
        # 根节点左子树置为空
        root.left = None


        # 让root指向当前右子树的最后结点
        while(root.right):
            root = root.right
        # 将前面临时保存已经展开的右子树继续接入
        root.right = temp
```

#### 138.数组异或操作(1486-Easy)

题意：给你两个整数，n 和 start 。数组 nums 定义为：nums[i] = start + 2*i（下标从 0 开始）且 n == nums.length 。请返回 nums 中所有元素按位异或（XOR）后得到的结果。

分析：依据题意模拟，无需构造数组，将数组中的值累异或到res中即可

```python
class Solution:
    def xorOperation(self, n: int, start: int) -> int:
        # 依据题意模拟-简化版
        res = start
        for i in range(1,n):
            res ^= start + 2 * i
        return res
```

#### 139.制作m束花所需的最少天数(1482-Medium)

##### Key：二分查找

题意：给你一个整数数组 bloomDay，以及两个整数 m 和 k 。现需要制作 m 束花。制作花束时，需要使用花园中 相邻的 k 朵花 。花园中有 n 朵花，第 i 朵花会在 bloomDay[i] 时盛开，恰好可以用于一束花中。请你返回从花园中摘 m 束花需要等待的最少的天数。如果不能摘到 m 束花则返回 -1 。

分析：采用二分查找寻找制作m束花需要最少的天数，天数最小值为数组花园中每朵花盛开天数的最小值，天数最大值为数组花园中每朵花盛开天数的最大值，即一朵花制作一束花要看花开的最小天数和最大天数，二分查找过程中不断判断当前天数day是否可以满足制作m束花，如果满足则移动右指针去判断更少的天数寻找最小值，如果不满足则移动右指针去寻找满足的最小值。单独设置当前天数day是否满足制作m束花函数，遍历数组花园依次判断小于传入天数的值累加花朵满足k朵则可以制作一束花，花束统计个数加一花朵数归零，当花束个数满足m时跳出循环返回。

```python
class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
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
    def checkDays(self, bloomDay: List[int], day: int, m: int, k: int) -> int:
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
```

#### 140.在D天内送达包裹的能力(1011-Meidum)

##### Key：二分查找

题意：传送带上的包裹必须在 D 天内从一个港口运送到另一个港口。传送带上的第 i 个包裹的重量为 weights[i]。每一天，我们都会按给出重量的顺序往传送带上装载包裹。我们装载的重量不会超过船的最大运载重量。返回能在 D 天内将传送带上的所有包裹送达的船的最低运载能力。

分析：这个题类似于上一道制作m束花需要的最少天数，二分查找运载能力寻找最低，设置函数判断以当前运载能力在D天能否将全部包裹运输完成，连续累加重量与运载能力比较，如果小于则在一天内可运输继续累加，如果大于说明需要额外增加天数，并且将当前累加重量置为当前包裹重量，因为目前超载需要移除当前这个放到第二天，根据判断结果从而移动左右指针寻找最低运载能力。

```python
class Solution:
    def shipWithinDays(self, weights: List[int], D: int) -> int:
        # 二分查找运送所有包裹最低运载能力，最低为weights中最大值，最高为weights的和
        left = max(weights)
        right = sum(weights)

        while left < right:
            # 先判断mid运载能力
            mid = (left + right) // 2
            # 检查以该运载能力能否在D天运输完成，可以的话移动右指针继续判断更小运载能力
            if self.checksDays(weights, D, mid):
                right = mid
            # 该运载能力在D天不能完成运输的话移动左指针判断更大运载能力
            else:
                left = mid + 1
        return left

    # 检查以capability作为运载能力在D天内能否运送完所有包裹
    def checksDays(self, weights: List[int], D: int, capability:int) -> int:
        current = 0
        day = 1
        for w in weights:
            current += w
            # 逐渐累加重量判断是否超过当前运载能力，如果超过则需要新的一天，不超过则可在1天内
            if current > capability:
                day += 1
                # 当前天的重量需要放到第二天
                current = w
        # 以当前运载能力全部运送完需要的天数与D比较，小于D说明当前能力可以，大于D说明当前不行
        return day <= D
```

#### 141.叶子相似的树(872-Easy)

##### Key：DFS深度优先遍历

题意：请考虑一棵二叉树上所有的叶子，这些叶子的值按从左到右的顺序排列形成一个 *叶值序列* 。如果有两棵二叉树的叶值序列是相同，那么我们就认为它们是 叶相似 的。如果给定的两个根结点分别为 root1 和 root2 的树是叶相似的，则返回 true；否则返回 false 。

分析：DFS递归先序遍历，传入根节点和结果数组，判断是叶子节点就将其值累加进结果数组中，递归遍历其左右子树。

```python
class Solution:
    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        # 深度优先遍历：传入根节点和结果数组
        def dfs(root: TreeNode, res: List[int]) -> None:
            if not root:
                return 
            # 叶子节点累加其值到结果数组中
            if root.left == None and root.right == None:
                res.append(root.val)
            # 递归遍历左子树
            dfs(root.left, res)
            # 递归遍历右子树
            dfs(root.right, res)

        # 依次求取两个子树叶子节点值集，比较是否相等
        res1 = []
        dfs(root1, res1)
        res2 = []
        dfs(root2, res2)
        # 比较两个结果数组是否相等
        return res1 == res2
```

#### 142.删除并获得点数(740-Meidum)

##### Key：动态规划

题意：给你一个整数数组 nums ，你可以对它进行一些操作。每次操作中，选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。之后，你必须删除每个等于 nums[i] - 1 或 nums[i] + 1 的元素。开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。

分析：将问题转换为打家劫舍问题，数组转成每家的点数，用其值作为新数组的下标，累加和作为新数组的值，即不能取相邻的

```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        # 将数组转换成每家的点数，转换成打家劫舍问题，不能取相邻的
        # [3,4,2]转换成[0,0,2,3,4]
        # [2,2,3,3,3,4]转换成[0,0,4,9,4]

        # 转换数组的长度取决于原数组中的最大值
        trans_len = max(nums) + 1
        trans = [0] * trans_len

        # 累加每家的点数
        for i in range(len(nums)):
            trans[nums[i]] += nums[i]
        
        # dp[i]表示经过操作可以获得的最大点数
        dp = [0] * trans_len
        dp[0] = trans[0]
        dp[1] = max(trans[0], trans[1])

        for i in range(2, trans_len):
            dp[i] = max(trans[i] + dp[i-2], dp[i-1])
    
        return dp[-1]
```

#### 143.子数组异或查询(1310-Medium)

##### Key:前缀异或

题意：有一个正整数数组 arr，现给你一个对应的查询数组 queries，其中 queries[i] = [Li, Ri]。对于每个查询 i，请你计算从 Li 到 Ri 的 XOR 值（即 arr[Li] xor arr[Li+1] xor ... xor arr[Ri]）作为本次查询的结果。并返回一个包含给定查询 queries 所有结果的数组。

分析：构建前缀异或数组，这句话就是精髓了over

```python
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        # 构造前缀异或累计数组mp，mp[i] 是所有 0 到 i 元素的与或的结果
        N = len(arr)
        # mp[0] = 0^arr[0] = arr[0]
        cur = 0 
        mp = []
        for i in range(N):
            # 累异或结果
            cur ^= arr[i]
            mp.append(cur)
        
        res = []
        # 遍历queries中每组元素的两个位置
        for l, r in queries:
            # 如果首位置为0，直接返回末位置前缀和
            if l == 0:
                res.append(mp[r])
            # 首位置-1的异或前缀 异或 末位置异或前缀结果，相同异或为0消除
            # 例如[1,3,4,8]，求(1,2)即为mp[0] ^ mp[2]，即1 ^ 1^3^4 = 3^4
            else:
                res.append(mp[l-1]^mp[r])
        return res
```

#### 144.不同的二叉搜索树（96-Meidum）

##### Key：动态规划

题意：给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？返回满足题意的二叉搜索树的种数。

分析：状态定义为`dp[i]表示由i个结点构成二叉搜索树的个数`，注意空树也是二叉树的一种，所以dp[0]=1，遍历所有1-n节点，分别计算当i作为根节点时对应的二叉树数量并求和，每种根节点对应的二叉树数量为其左子树数量乘以右子树数量。

```python
class Solution:
    def numTrees(self, n: int) -> int:
        # dp[i]表示由i个结点构成二叉搜索树的个数,空树也是一种，所以dp[0]=1
        dp = [0] * (n+1)
        dp[0] = 1

        # 节点个数从1到n的所有情况
        for i in range(1, n+1):
            # 每种根节点对应的二叉树数量并求和
            for j in range(i):
                # 每种根节点对应的二叉树数量为其左子树数量乘以右子树数量
                dp[i] += dp[j] * dp[i-j-1]
        
        return dp[n]
```

#### 145.整数转罗马数字（12-Meidum）

题意：罗马数字包含以下七种字符： `I`， `V`， `X`， `L`，`C`，`D` 和 `M`。

|  I   |  1   |
| :--: | :--: |
|  V   |  5   |
|  X   |  10  |
|  L   |  50  |
|  C   | 100  |
|  D   | 500  |
|  M   | 1000 |

例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给你一个整数，将其转为罗马数字。

分析：先将表格内+一些小数在大数左边的特殊情况列到哈希表中，遍历哈希表然后判断num是否整除，当遇到num可以整除的key时，说明遇到了可以用来表示num的最大数字，则整除看需要几个key加入结果中，然后num对key取余继续判断后面。

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        # 使用哈希表，按照从大到小顺序排列
        hashmap = {1000:'M', 900:'CM', 500:'D', 400:'CD', 100:'C', 90:'XC', 50:'L', 40:'XL', 10:'X', 9:'IX', 5:'V', 4:'IV', 1:'I'}
        res = ''
        for key in hashmap:
            # 比当前数字小的最大值可用来表示num
            if num // key != 0:
                # 计算有几个key
                count = num // key
                # 加入结果中
                res += hashmap[key] * count
                # key用完后减少num
                num %= key
        return res
```

#### 146.罗马数字转整数（13-Easy）

题意：前面的题目内容和前一题相同，是上一题的逆过程将罗马数字转换成整数

分析：罗马数字转整数较为简单，注意判断特殊的三种情况，判断一下当前字符表示的数字是否小于下个字符表示的数字，如果小于的话则要在结果基础上减去当前字符表示的数字，否则正常情况就累加。注意如果是IV的话，res不是直接等于V-I，是先变成-I，然后遍历到V的时候+V。

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        map = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
        res = 0
        for i in range(len(s)):
            # 小的数字在大的数字左边特殊情况
            if i < len(s) - 1 and map[s[i]] < map[s[i+1]]:
                # 如果是IV这种，在遍历到I的时候res是-I，遍历到V的时候是加V，所以最后res是V-I
                res -= map[s[i]]
            # 遍历最后一个数字或者当前数字大于后面数字的时候累加
            else:
                res += map[s[i]]
        return res
```

#### 148.判断子序列（392-Easy)

题意：给定字符串 s 和 t ，判断 s 是否为 t 的子序列。字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

分析：本来在动态规划标签下找到的，发现这个题并不是非常适合动态规划，所以感觉双指针的方法还是挺好的还简单易懂都不需要我多解释了哈哈，就遍历匹配移动判断结束。

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        m = len(s)
        n = len(t)
        # 定义双指针，i遍历s，j遍历t
        i,j = 0,0
        while i < m and j < n:
            # 当字符匹配上之后移动i继续判断s中下一个字符
            if s[i] == t[j]:
                i += 1
            # 如果没有匹配上或者当前匹配完成则j向后移动
            j += 1
        # 最后判断i是否移动到s串末尾
        return i == m
```

#### 149.形成两个异或相等数组的三元组数目(1442-Meidum)

题意：给你一个整数数组 arr 。现需要从数组中取三个下标 i、j 和 k ，其中 (0 <= i < j <= k < arr.length) 。a 和 b 定义如下：a = arr[i] ^ arr[i + 1] ^ ... ^ arr[j - 1]，b = arr[j] ^ arr[j + 1] ^ ... ^ arr[k]
注意：^ 表示 按位异或 操作。请返回能够令 a == b 成立的三元组 (i, j , k) 的数目。

分析：不要去做分割，注意利用异或的性质相同的数异或结果为0，所以[i,k]数组累异或结果为0的话，j可以是其中任意一个位置除i外，所以结果数目累加k-i，所以求解累异或为0的[i,k]数组。

```python
class Solution:
    def countTriplets(self, arr: List[int]) -> int:
        # [i,k]如果满足条件即累异或和应为0，因为由j分开两个相同数异或为0
        N = len(arr)
        ans = 0

        # i不能取到最后一位
        for i in range(N-1):
            # 计算[i,k]的累异或
            sum = 0
            for k in range(i,N):
                sum ^= arr[k]
                # 如果累异或为零，说明[i,k]除了i以外均可以当做j来分割
                if sum == 0:
                    # j不能取i可以取到k，所以是k-i
                    ans += (k - i)
        return ans
```

#### 150.盛最多水的容器(11-Meidum)

题意：给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。说明：你不能倾斜容器。

分析：利用双指针，移动中计算围成的区域面积，高度由最短的板决定，不断更新面积最大值。

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        # 最优做法：双指针

        # 设置左右指针，边移动边判断
        left = 0
        right = len(height) - 1
        maxArea = 0

        while left < right:
            # 先计算长度
            w = right - left
            # 计算高度：查看左右的高度，储水量取决于最低的木板，计入h后移动指针
            if height[left] < height[right]:
                h = height[left]
                left += 1
            else:
                h = height[right]
                right -= 1
            
            # 计算当前区域面积
            area = w * h
            maxArea = max(area, maxArea)
        return maxArea
```

#### 151.找出第K大的异或坐标值(1738-Meidum)

题意：给你一个二维矩阵 matrix 和一个整数 k ，矩阵大小为 m x n 由非负整数组成。矩阵中坐标 (a, b) 的 值 可由对所有满足 0 <= i <= a < m 且 0 <= j <= b < n 的元素`matrix[i][j]`（下标从 0 开始计数）执行异或运算得到。请你找出 matrix 的所有坐标中第 k 大的值（k 的值从 1 开始计数）。

分析：求二维的前缀异或和，注意初始化多一行多一列，画出矩阵出来会好理解。

```python
class Solution:
    def kthLargestValue(self, matrix: List[List[int]], k: int) -> int:
        
        m = len(matrix)
        n = len(matrix[0])
        res = []

        # 将首行和首列空出来赋予默认值 0，并使用接下来的 m 行和 n 列存储二维前缀和
        # pre[i][j]表示(i-1,j-1)的值，即从(0,0)到(i-1,j-1)异或和
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1,m+1):
            for j in range(1,n+1):
                # 计算前缀异或和，小于i等于j ^ 小于j等于i ^ 小于i又小于j，最后异或matrix当前元素
                pre[i][j] = pre[i-1][j] ^ pre[i][j-1] ^ pre[i-1][j-1] ^ matrix[i-1][j-1]
                res.append(pre[i][j])
        # 对结果列表降序排序，取第k-1表示第k个最大值返回
        res.sort(reverse=True)
        return res[k-1]
```

#### 152.二叉树的层序遍历(102-Medium)

题意：给你一个二叉树，请你返回其按 **层序遍历** 得到的节点值。 （即逐层地，从左到右访问所有节点）。

分析：利用队列存储每一层的结点，然后遍历队列将结点值添加到结果列表中。

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        # 根节点不存在特殊情况判断
        if not root: return []

        # 利用队列保存每一层的结点
        queue = [root]
        # 结果列表
        res = []

        while queue:
            # 遍历上一层结点的值以列表形式存储在res
            res.append([node.val for node in queue])
            # 存储当前层结点
            temp = []
            for node in queue:
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            # 将queue变成当前层结点
            queue = temp
        return res
```

#### 153.二叉树的直径(543-Easy)

题意：给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。

分析：这个问题可以转化为求每个节点的左右节点的最大深度之和的最大值，注意如果设置全局变量的话，要在变量名之前加self，即可在其他函数内部改变。本题构造一个深度优先遍历函数求解左右子树的最大深度，并在函数中实时更新最大深度相加之和。

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        # 变量加上self即可在其他函数中也可以改变
        self.res = 0
        self.maxDeep(root)
        return self.res
    
    # 深度优先遍历求解：根节点对应左右结点最大深度之和的最大值
    def maxDeep(self, root: TreeNode) -> int:
        if not root:
            return 0
        # 遍历得到左子树的最大深度
        dl = self.maxDeep(root.left)
        # 遍历得到右子树的最大深度
        dr = self.maxDeep(root.right)
        # 求解深度之和的最大值
        self.res = max(dl+dr, self.res)
        # 返回依次为根节点所对应的最大深度
        return max(dl, dr) + 1
```

#### 154.前K个高频单词(692-Meidum)

##### Key：哈希表结合优先队列

题意：给一非空的单词列表，返回前 *k* 个出现次数最多的单词。返回的答案应该按单词出现频率由高到低排序。如果不同的单词有相同出现频率，按字母顺序排序。

分析：首先想起来之前做过的前K个高频元素，然后移动了一下哈希表和最小堆的方法，这个单词特殊的地方在于，出现次数相同的要按照首字母排序，而不是前后的相对顺序。所以在用哈希表获取到key和value之后，先进行一波排序，按照先value降序后key升序构成新列表，然后再经过最小堆，最后再对最小堆进行一次调整。

```python
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # 先统计每个单词出现的次数
        count = collections.Counter(words)
        # 按照value降序key升序排序形成新列表，（出现次数相同按照单词首字母排序小的要在前面）
        count1 = sorted(count.items(), key = lambda x :(-x[1], x[0]))

        # 构造小顶堆
        heap = []
        # 遍历键值对，存储到堆中，堆中保持k个元素
        for key,val in count1:
            # 当此时堆中已有k个元素或者大于k个元素时，
            if  len(heap) >= k:
                # 比较当前值与堆顶，大于堆顶则弹出堆顶将新元素压入堆中，保证堆内是前k个最大值
                if val > heap[0][0]:
                    heapq.heapreplace(heap, (-val,key))
            # 当前堆小于k个元素继续入堆
            else:
                heapq.heappush(heap, (val, key))
        
        # 将小顶堆按照val降序排序，按key升序
        heap.sort(key=lambda x:(-x[0], x[1]))
        # 输出堆内元素的第二维key
        return [item[1] for item in heap]
```

#### 156.从前序与中序遍历序列构造二叉树(105-Medium)

题意：根据一棵树的前序遍历与中序遍历构造二叉树。

分析：先序序列是根节点，左子树，右子树，中序序列是左子树，根节点，右子树。

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:

    # 找到根在中序中的索引loc,中序中loc，左边是root.left 右边是root.right，所以inorder[:loc]
    # 前序遍历中第一个元素是root,  root.left 有 loc个，所以 preorder[1: loc + 1]
        
        if not preorder: return None

        # 前序遍历的首元素作为根节点
        val = preorder[0]
        node = TreeNode(val)

        # 寻找根节点在中序遍历中的位置
        loc = inorder.index(val)

        node.left = self.buildTree(preorder[1:loc+1], inorder[:loc])
        node.right = self.buildTree(preorder[loc+1:], inorder[loc+1:])

        return node
```

#### 157.电话号码的字母组合(17-Meidum)

##### Key：回溯法

题意：给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

分析：看到所有组合应用回溯法，定义字典和回溯函数，当nextdigit非空时，对于 nextdigit[0] 中的每一个字母 letter，执行回溯，直至 nextdigit 为空。最后将 path 加入到结果中。

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:

        if not digits: return []
        # 构建字典
        phone = {'2':['a','b','c'],
                 '3':['d','e','f'],
                 '4':['g','h','i'],
                 '5':['j','k','l'],
                 '6':['m','n','o'],
                 '7':['p','q','r','s'],
                 '8':['t','u','v'],
                 '9':['w','x','y','z']}

        # 定义回溯函数，当nextdigit非空时，对于 nextdigit[0] 中的每一个字母 letter，执行回溯
        def backtrace(path, nextdigits):
            # 直至 nextdigit 为空。最后将 path 加入到结果中。
            if len(nextdigits) == 0:
                res.append(path)
            else:
                for word in phone[nextdigits[0]]:
                    backtrace(path + word, nextdigits[1:])
        res = []
        backtrace('', digits)
        return res
```

#### 158.4的幂（342-Easy）

题意：给定一个整数，写一个函数来判断它是否是 4 的幂次方。如果是，返回 true ；否则，返回 false 。

整数 n 是 4 的幂次方需满足：存在整数 x 使得 n == 4x

分析：4的幂首先一定先是2的幂，然后就再判断是否是4的幂，可以利用奇数位校验

```python
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        # 首先判断是否为2的幂
        if n < 0 or n & (n-1):
            return False
            
        # 判断n是否为4的幂，4的n即为(3+1)的n，展开多项式除结尾的1都有3相乘，所以4的幂一定可以模3余1
        # return n % 3 == 1

        # 或者判断二进制的1是否在奇数位，奇数位校验
        return n & 0x55555555 > 0
```

#### 159.你能在你最喜欢的那天吃到你最喜欢的糖果吗？（1744-Meidum）

##### Key：前缀和

题意：给你一个下标从 0 开始的正整数数组 candiesCount ，其中 candiesCount[i] 表示你拥有的第 i 类糖果的数目。同时给你一个二维数组 queries ，其中 queries[i] = [favoriteTypei, favoriteDayi, dailyCapi] 。

你按照如下规则进行一场游戏：你从第 0 天开始吃糖果。
你在吃完 所有 第 i - 1 类糖果之前，不能 吃任何一颗第 i 类糖果。
在吃完所有糖果之前，你必须每天 至少 吃 一颗 糖果。
请你构建一个布尔型数组 answer ，满足 answer.length == queries.length 。answer[i] 为 true 的条件是：在每天吃 不超过 dailyCapi 颗糖果的前提下，你可以在第 favoriteDayi 天吃到第 favoriteTypei 类糖果；否则 answer[i] 为 false 。注意，只要满足上面 3 条规则中的第二条规则，你就可以在同一天吃不同类型的糖果。请你返回得到的数组 answer

分析：六一儿童节快乐！别的不多说了

```python
class Solution:
    def canEat(self, candiesCount: List[int], queries: List[List[int]]) -> List[bool]:

        n = len(candiesCount)
        # candiesCount前缀和:得到想把前i种水果吃完需要吃多少个，前i-1类糖果总和
        preSum = [0] * (n+1)
        for i in range(n):
            preSum[i+1] = preSum[i] + candiesCount[i]
        
        m = len(queries)
        ans = [False] * m
        for i, (t,d,limit) in enumerate(queries):
            # 到favoriteDayi可以吃到的最小糖果数
            min_candy = d + 1
            # 到favoriteDayi可以吃到的最大糖果数
            max_candy = limit * (d + 1)
            # 尽量少吃不能吃到下一种，尽量多吃不能小于前一种
            if min_candy <= preSum[t+1] and max_candy > preSum[t]:
                ans[i] = True
        return ans
```

#### 160.反转每对括号间的子串（1190-Meidum）

题意：给出一个字符串 s（仅含有小写英文字母和括号）。

请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

注意，您的结果中 不应 包含任何括号。

分析：逐层翻转，将翻转结果放回原位置，再和外层继续翻转

```python
class Solution:
    def reverseParentheses(self, s: str) -> str:
        # 题意：先翻转内层，翻转结果还放回原位置和外层继续翻转
        # (ed(et(oc))el)" -> etco -> octe - > leetcode
        stack = []
        for c in s:
            # 暂时存储待反转的字符串
            temp = []
            # 没遇到右括号均压入栈中
            if c != ')':
                stack.append(c)
            else:
                # 遇到右括号，将当前括号所有值弹出存入temp
                while stack and stack[-1] != '(':
                    temp.append(stack.pop())
                # 再将左括号去除
                stack.pop()
                # 将逆序后的字符串加回栈中，等待下一次逆序（用+=加回在栈顶的末尾位置）
                stack += temp
        # 最后栈内即为结果
        return "".join(stack)
```

#### 161.连续数组(525-Meidum)

##### Key:前缀和

题意：给定一个二进制数组 `nums` , 找到含有相同数量的 `0` 和 `1` 的最长连续子数组，并返回该子数组的长度。

分析：先将数组中的0转换为-1，这样寻找含有相同数量0和1的最长连续子数组转换成寻找和为0的最长连续子数组，计算前缀和，同时将前缀和和第一次出现的下标存储到哈希表中，在计算后面前缀和的时候判断是否在哈希表中出现过，如果出现过说明存在两个前缀和相等，说明其区间连续数组和为0，计算数组长度及时更新结果。

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        # 前缀和 + 哈希表

        # 将0转换为-1，相同0和相同1的个数的子数组将0换成-1之后和为0
        N = len(nums)
        for i in range(N):
            if nums[i] == 0:
                nums[i] = -1
        
        # 简历哈希表存储每个preSum[i]出现的最早下标，
        # 如果preSum[i]==preSum[j]，即说明i~j的连续子数组和为0
        map = {0:0}
        res = 0

        preSum = [0] *(N+1)
        for i in range(1, N+1):
            preSum[i] = preSum[i-1] + nums[i-1]
            # 判断是否出现相等的前缀和，出现的话计算下标之间的子数组长度更新res
            if preSum[i] in map:
                res = max(i - map[preSum[i]], res)
            else:
                map[preSum[i]] = i

        return res
```

#### 162.相交链表（160-Easy）

题意：给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 `null` 。

分析：当 pA 到达链表的尾部时，将它重定位到链表 B 的头结点；同样的，当 pB 到达链表的尾部时，将它重定位到链表 A 的头结点。若在某一时刻 pA 和 pB 相遇，则 pA/pB 为相交结点。两个链表相交，则相交后的长度是相同的，我们需要让两个链表从同距离末尾同等距离的位置开始遍历。这个位置只能是较短链表的头结点位置。为此必须消除两个链表的长度差

示例：listA = [1,3,5,7,9,11], listB = [2,4,9,11] 相交的节点为 9
链表 A 的长度为 6；链表 B 的长度为 4。
pB 比 pA 少经过 2 个结点，会先到达尾部。此时将 pB 指向链表 A 的头结点，当 pA 到达链表 A 的尾部时，将 pA 指向链表 B 的头结点，pB 要比 pA 多走 2 个结点。因此，它们会同时到达交点。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA is None or headB is None:
            return None
        # 双指针
        left, right = headA, headB
        # 两个指针走相同的路消除长度差，可以同时到达终点
        while left != right:
            # 如果先走到链表尾部，移动指针到另一链表头结点
            left = left.next if left else headB
            right = right.next if right else headA
        return left
```

#### 163.移除链表元素(203-Easy)

题意：给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 。

分析：这个题存在可能会删除头结点的情况，所以需要设立一个新结点作为链表头结点，将原头结点链接在后面作为普通结点处理。

```python
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:

        # 创建新结点作为链表头结点，方便将原head节点当成普通结点操作，可以进行删除
        new_head = ListNode(-1)
        # 新头结点指向原头结点
        new_head.next = head 
        # 记录新头结点，从.next开始遍历
        node = new_head

        # 开始遍历寻找删除值
        while node.next:
            if node.next.val == val:
                node.next = node.next.next
            else:
                node = node.next
        
        return new_head.next
```

#### 167.三数之和(15-Medium)

##### Key:三指针

题意：给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。注意：答案中不可以包含重复的三元组。

分析：首先排序，第一个指针用来枚举第一个元素，时刻保证不会有重复，双指针从first+1到n-1开始向中间遍历，寻找和等于-nums[first]，如果数值过大右指针左移，如果数值过小左指针右移，如果数值合适做出决策添加结果，然后移动左右指针，注意去除重复。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 三指针
        n = len(nums)
        nums.sort()
        res = []
        # 枚举第一个元素
        for first in range(n-2):
            # 数组里最小的都大于0 不可能有答案
            if nums[first] > 0: break
            # 保证first不会有重复
            if first > 0 and nums[first] == nums[first-1]:continue
            # 标准双指针写法
            second, third = first + 1, n - 1
            while second < third:
                target = -nums[first]
                sum = nums[second] + nums[third]
                # 当前数值太大 做出决策：右指针左移
                if sum > target:
                    third -= 1
                # 当前数值太大 做出决策：右指针左移
                elif sum < target:
                    second += 1
                # 数值正合适 做出决策：左指针右移且右指针左移 注意不能重复
                else:
                    res.append([nums[first], nums[second], nums[third]])
                    second += 1
                    third -= 1
                    while third > second and nums[third] == nums[third+1]: third -= 1
                    while third > second and nums[second] == nums[second-1]: second += 1
        return res
```

#### 168.排序链表(148-Meidum)

题意：给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

分析：开始用列表的方法将链表中数据取出来排序，其实应用用链表的排序方法来操作。这里主要分三步，首先利用快慢指针找到链表中点，然后断链，最后合并。

```python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # 递归归并排序
        # 通过快慢指针找到链表中点-断链-合并
        # 1.通过快慢指针找到链表中点
        if head is None or head.next is None:
            return head
        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # 2. 断链 + 递归排序
        rightHead = slow.next
        slow.next = None
        left = self.sortList(head)
        right = self.sortList(rightHead)
        # 3. 迭代合并
        return self.mergeTwoLists(left, right)


    # 3.迭代合并
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        newHead = ListNode(-1)
        cur = newHead
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 if l1 else l2
        return newHead.next
```

#### 169.串联字符串的最大长度（1239-Medium）

##### Key：回溯法

题意：给定一个字符串数组 arr，字符串 s 是将 arr 某一子序列字符串连接所得的字符串，如果 s 中的每一个字符都只出现过一次，那么它就是一个可行解。请返回所有可行解 s 中最长长度。

分析：回溯法依次拼接arr中的字符，利用set判断是否含有重复字符，没有的话记录最长长度。

```python
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        # 回溯法
        self.res = 0
        self.backtrace(0, arr, '')
        return self.res
    
    def backtrace(self, index, arr, temp):
        # 判断temp中是否含有重复字符，没有的话记录最长长度
        if len(temp) == len(set(temp)):
            self.res = max(self.res, len(temp))
        # 回溯
        for i in range(index, len(arr)):
            self.backtrace(i+1, arr, temp + arr[i])
```

#### 171.环形链表(141-Easy)

题意：给定一个链表，判断链表中是否有环。如果链表中存在环，则返回 `true` 。 否则，返回 `false` 。

分析：利用快慢指针判断是否存在环，如果存在环的话则快指针会一直在环里面绕，如果慢指针追上快指针则快慢指针会相遇。

```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        # 快慢指针判断是否存在环，如果快慢指针相遇则存在环
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

#### 172.二叉树的最近公共祖先(236-Medium)

##### Key：回溯

题意：给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

分析：递归回溯，分别从当前节点的左右子树中寻找最近公共祖先，判断返回的left和right是否都存在，如果存在则返回当前的root，有一方不存在则返回另一方。

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 递归回溯
        # 终止条件：当root节点为空，或遍历到叶子节点的子节点
        if not root:
            return
        # 当root == p或root == q时，即可终止
        if root == p or root == q:
            return root
        # 从当前节点root的左子树中寻找最近公共祖先
        left = self.lowestCommonAncestor(root.left, p, q)
        # 从当前节点root的右子树中寻找最近公共祖先
        right = self.lowestCommonAncestor(root.right, p, q)
        # 如果left和right都存在，表示他们异侧
        if left and right:
            return root
        # 如果有一个为空则返回另一个
        elif left and not right:
            return left
        elif right and not left:
            return right
```

#### 173.把二叉搜索树转换为累加树（538-Medium）

题意：给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。

分析：一个图画的很不错的题解：https://leetcode-cn.com/problems/convert-bst-to-greater-tree/solution/shou-hua-tu-jie-zhong-xu-bian-li-fan-xiang-de-by-x/，反向中序遍历：访问的结点值递减，每次累加到num。

```python
class Solution:
    def __init__(self):
        # num 始终保存‘比当前节点值大的所有节点值的和’
        self.num = 0
    # 反向中序遍历：访问的节点值是递减的，之前访问的节点值都比当前的大，每次累加给 num 即可
    def convertBST(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        # 递归右子树
        self.convertBST(root.right)
        # 处理当前节点
        root.val += self.num
        self.num = root.val
        # 递归左子树
        self.convertBST(root.left)
        return root
```

#### 174.除自身以外数组的乘积(238-Medium)

题意：给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。

分析：我们不必将所有数字的乘积除以给定索引处的数字得到相应的答案，而是利用索引左侧所有数字的乘积和右侧所有数字的乘积（即前缀与后缀）相乘得到答案。对于给定索引 i，我们将使用它左边所有数字的乘积乘以右边所有数字的乘积。初始化两个空数组存储左积和右积，利用两个循环来填充数组的值，填充完成后在输入数组上迭代计算二者相乘。

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # 左右乘积列表：利用索引左侧所有数字乘积和右侧所有数字乘积（即前缀与后缀）相乘得到答案
        N = len(nums)
        left, right, res = [0] * N, [0] * N, [0] * N

        # 索引为0的元素左侧没有元素，因此初始化为1
        left[0] = 1
        for i in range(1, N):
            left[i] = left[i-1] * nums[i-1]
        
        # 索引为N-1的元素右侧没有元素，因此初始化为1
        right[N-1] = 1
        for i in range(N-2, -1, -1):
            right[i] = right[i+1] * nums[i+1]
        
        # 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
        for i in range(N):
            res[i] = left[i] * right[i]
        
        return res
```

#### 175.二叉树的最大深度(104-Easy)

题意：给定一个二叉树，找出其最大深度。二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

分析：递归

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1
```

#### 176.任务调度器（621-Meidum）

题意：给你一个用字符数组 tasks 表示的 CPU 需要执行的任务列表。其中每个字母表示一种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。在任何一个单位时间，CPU 可以完成一个任务，或者处于待命状态。然而，两个 相同种类 的任务之间必须有长度为整数 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。你需要计算完成所有任务所需要的 最短时间 。

分析：参考这篇题解：https://leetcode-cn.com/problems/task-scheduler/solution/tong-zi-by-popopop/

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
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

```

#### 177.雪糕的最大数量(1833-Meidum)

##### Key：贪心

题意：夏日炎炎，小男孩 Tony 想买一些雪糕消消暑。商店中新到 n 支雪糕，用长度为 n 的数组 costs 表示雪糕的定价，其中 costs[i] 表示第 i 支雪糕的现金价格。Tony 一共有 coins 现金可以用于消费，他想要买尽可能多的雪糕。给你价格数组 costs 和现金量 coins ，请你计算并返回 Tony 用 coins 现金能够买到的雪糕的 最大数量 。注意：Tony 可以按任意顺序购买雪糕。

分析：看起来好像是背包问题，由于数据范围的问题所以要用贪心。因为要买雪糕的最大数量，所以要优先买价格低的，先将数组排序，然后按照顺序买。

```python
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        # 贪心 + 排序
        # 优先买价格低的，将数组排序，直到coins花完
        costs.sort()
        for i in range(len(costs)):
            # 如果钱花完但是雪糕还有，就直接返回i，因为循环已经进入到买不起的那个了
            if coins < costs[i]:return i
            coins -= costs[i]
        # 如果都买得起说明costs数组遍历完，返回数组长度i+1
        return i + 1
```

#### 178.根据字符出现频率排序(451-Medium)

题意：给定一个字符串，请将字符串里的字符按照出现的频率降序排列。

分析：将字符串转为可迭代的对象创建计数器，然后计数完成之后再转回可迭代的对象按照值进行排序，然后按照出现次数将结果字符串再拼接出来。

```python
class Solution:
    def frequencySort(self, s: str) -> str:
        # 先将字符串转为可迭代的对象创建计数器
        str_list = list(s)
        countMap = collections.Counter(str_list)
        # 再将计数器转为可迭代的键值对元组形式根据值从大到小逆序排序
        char_tuple = countMap.items()
        res_tuple = sorted(char_tuple, key=lambda item:item[1],reverse = True)
        # 按照每一组出现的频率数将字符拼接起来
        res = ""
        for item in res_tuple:
            for _ in range(item[1]):
                res += item[0]
        return res
```

#### 179.错误的集合(645-Easy)

题意：集合 s 包含从 1 到 n 的整数。不幸的是，因为数据错误，导致集合里面某一个数字复制了成了集合里面的另外一个数字的值，导致集合 丢失了一个数字 并且 有一个数字重复 。给定一个数组 nums 代表了集合 S 发生错误后的结果。请你找出重复出现的整数，再找到丢失的整数，将它们以数组的形式返回。

分析：利用哈希表存储下标为键，出现次数为值。寻找出现两次和出现一次的下标加1就是需要返回的结果。

```python
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        # Hash：下标为键，出现个数为值
        hash_list = [0] * len(nums)
        for num in nums:
            hash_list[num - 1] += 1
        return [hash_list.index(2) + 1, hash_list.index(0) + 1]
```

#### 180.原子的数量(726-Hard)

题意：给定一个化学式formula（作为字符串），返回每种原子的数量。原子总是以一个大写字母开始，接着跟随0个或任意个小写字母，表示原子的名字。如果数量大于 1，原子后会跟着数字表示原子的数量。如果数量等于 1 则不会跟数字。例如，H2O 和 H2O2 是可行的，但 H1O2 这个表达是不可行的。两个化学式连在一起是新的化学式。例如 H2O2He3Mg4 也是化学式。一个括号中的化学式和数字（可选择性添加）也是化学式。例如 (H2O2) 和 (H2O2)3 是化学式。给定一个化学式，输出所有原子的数量。格式为：第一个（按字典序）原子的名子，跟着它的数量（如果数量大于 1），然后是第二个原子的名字（按字典序），跟着它的数量（如果数量大于 1），以此类推。

分析：官方题解：https://leetcode-cn.com/problems/number-of-atoms/solution/yuan-zi-de-shu-liang-by-leetcode-solutio-54lv/

```python
class Solution:
    def countOfAtoms(self, formula: str) -> str:
        # 与官方题解的思想相同
        stack = [defaultdict(int)]
        n = len(formula)
        i = 0
        num = 0
        last_atom = None
        while i < n:
            if '0' <= formula[i] <= '9':
                num = 10 * num + int(formula[i])
                i += 1
            elif 'A' <= formula[i] <= 'Z':
                stack[-1][last_atom] += num if num > 1 else 1
                num = 0
                if i + 1 < n and 'a' <= formula[i + 1] <= 'z':
                    last_atom = formula[i:i+2]
                    i += 2
                else:
                    last_atom = formula[i:i+1]
                    i += 1
            elif formula[i] == '(':
                stack[-1][last_atom] += num if num > 1 else 1
                last_atom = None
                num = 0

                stack.append(defaultdict(int))
                i += 1
            elif formula[i] == ')':
                stack[-1][last_atom] += num if num > 1 else 1
                last_atom = None
                num = 0
                
                if i + 1 < n and '0' <= formula[i + 1] <= '9':
                    i += 1
                    mul_start = i
                    while i < n and '0' <= formula[i] <= '9':
                        i += 1
                    mul = int(formula[mul_start:i])
                else:
                    i += 1
                    mul = 1
                
                d = stack.pop()
                for atom in d:
                    stack[-1][atom] += d[atom] * mul
        
        # special case for the final one
        stack[-1][last_atom] += num if num > 1 else 1
        d = [(c, freq) for c, freq in stack.pop().items() if c]
        res = ''
        for c, freq in sorted(d):
            res += c 
            if freq > 1:
                res += str(freq)

        return res

```

#### 181.最佳买卖股票时机含冷冻期(309-Medium)

题意：给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

分析：状态方程定义：`dp[i][0]`: 第i天买入状态，所持最大金额；`dp[j][1]`: 第i天卖出状态，所持最大金额。状态转移方程：今天的买入状态, 可以 1. 维持昨天买入， 2. 前天卖出由今天买入 （买入有冷冻期约束）；两种情况取最大。今天的卖出状态, 可以 1. 维持昨天卖出， 2. 昨天买入由今天卖出 （卖出无冷冻期约束）；两种情况取最大。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # dp[i][0]表示第i天买入状态所持最大金额
        # dp[i][1]表示第i天卖出状态所持最大金额
        if len(prices) == 1:return 0
        dp = [[0, 0] for _ in range(len(prices))]
        # 第0天买入
        dp[0][0] = -prices[0]
        # 第一天买入 = max(维持昨天买入，前天卖出0 - 今天买入)
        dp[1][0] = max(-prices[0], -prices[1])
        # 第一天卖出 = max(维持昨天卖出，昨天买入今天卖出)
        dp[1][1] = max(dp[0][1], dp[0][0] + prices[1])
        for i in range(2, len(prices)):
            # 第i天买入 = max(维持昨天买入，前天卖出 - 今天买入（含冷冻期）)
            dp[i][0] = max(dp[i-1][0], dp[i-2][1] - prices[i])
            # 第i天卖出 = max(维持昨天卖出，昨天买入 + 今天卖出)
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        return dp[-1][-1]
```

#### 182.主要元素（面试题17.10-Easy）

##### Key：摩尔投票法

题意：数组中占比超过一半的元素称之为主要元素。给你一个 整数 数组，找出其中的主要元素。若没有，返回 -1 。请设计时间复杂度为 O(N) 、空间复杂度为 O(1) 的解决方案。

分析：空间复杂度为1的方法学习一种新的摩尔投票法，在每一轮投票过程中，从数组中删除两个不同的元素。维护主要候选元素和主要元素出现的次数，遍历nums判断元素是否与x相等，如果count不存在则赋予新的候选值，与x不相等count-1抵消一次，与x相等count+1，最后如果count存在则判断当前候选x出现次数是否大于数组长度的一半。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 摩尔投票法
        N = len(nums)
        x = -1
        count = 0
        # 遍历数组
        for num in nums:
            # count不存在给x赋予新的候选值
            if not count:
                x = num
            # 与x不同count-1相同count+1
            count += 1 if x == num else -1
        return x if count and nums.count(x) > N // 2 else -1
```

#### 183.H指数（274-Medium）

题意：给定一位研究者论文被引用次数的数组（被引用次数是非负整数）。编写一个方法，计算出研究者的 *h* 指数。数组长度表示论文篇数，数组内容表示每篇论文的引用次数。H指数的定义(来自维基百科）: H指数的计算基于其研究者的论文数量及其论文被引用的次数。赫希认为：一个人在其所有学术文章中有N篇论文分别被引用了至少N次，他的H指数就是N。

分析：这个题的题目说的不清晰，所以改了题意。H指数的定义是一个人的论文中有N篇论文分别被引用了至少N次，他的H指数就是N。论文数量即为下标加一，引用次数即为数组中的内容，先将数组逆序排序，因为H指数一定出现在数组的后半部分。依次判断引用次数是否大于等于论文数量，如果是的话记录一下当前的H指数（下标+1），实时更新最大的H指数。

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        # 维基百科：一个人在其所有学术文章中有N篇论文分别被引用了至少N次，他的H指数就是N
        # 逆序排序
        citations.sort(reverse = True)
        N = len(citations)
        res = 0
        # 从高到低判断当前引用数citations[i]是否大于等于(至少)当前论文数(i+1)，如果是的话则i+1记录为一个H指数
        for i in range(N):
            if citations[i] >= i + 1:
                res = max(res, i + 1)
            else:
                break
        return res
```

#### 184.减小和重新排列数组后最大元素(1846-Meidum)

##### Key:贪心

题意：给你一个正整数数组 arr 。请你对 arr 执行一些操作（也可以不进行任何操作），使得数组满足以下条件：arr 中 第一个 元素必须为 1 。任意相邻两个元素的差的绝对值 小于等于 1 ，也就是说，对于任意的 1 <= i < arr.length （数组下标从 0 开始），都满足 abs(arr[i] - arr[i - 1]) <= 1 。abs(x) 为 x 的绝对值。

你可以执行以下 2 种操作任意次：减小 arr 中任意元素的值，使其变为一个 更小的正整数 。重新排列 arr 中的元素，你可以以任意顺序重新排列。请你返回执行以上操作后，在满足前文所述的条件下，arr 中可能的最大值 。
分析：先排序，然后用贪心算法，限定首位为1，依次判断数组中的元素是否与上一位相差为1，如果相差为1则不变，相差大于1则把当前数改为仅比上一位数大1的数。

```python
class Solution:
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        # 排序之后用贪心
        arr.sort()
        # 限定首位为1
        arr[0] = 1
        # 依次判断数组中的元素是否与上一位仅相差为1，
        for i in range(1, len(arr)):
            # 相差小于等于1不变，相差大于1，则将当前数改成仅比上一位数大1的数
            if (arr[i] - arr[i-1]) > 1:
                arr[i] = arr[i-1] + 1
        # 遍历结束后最后的数就是数组中最大的数
        return arr[-1]
```

#### 185.单词拆分(139-Meidum)

##### Key:动态规划

题意：给定一个**非空**字符串 *s* 和一个包含**非空**单词的列表 *wordDict*，判定 *s* 是否可以被空格拆分为一个或多个在字典中出现的单词。

分析：dp[i]表示以i-1结尾的字符串是否可以被wordDict拆分，依次截取s中子串，当前一个子串为True且当前子串在字典中时，当前以i-1为结尾的字符串可以被拆分设置为True。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # dp[i]表示以i-1结尾的字符串是否可以被wordDict拆分
        N = len(s)
        dp = [False for _ in range(N + 1)]
        dp[0] = True
        
        for i in range(1, N + 1):
            for j in range(i):
                # dp[i]为true的前提是：dp[j]为true且j到i之间的单词在单词表中
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    # 一旦dp[i]为True了，后面的j也不用遍历
                    break
        return dp[-1]
```

#### 186.最短无序连续子数组(581-Meidum)

题意：给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。请你找出符合题意的最短子数组，并输出它的长度。

分析：正序遍历寻找右端点，记录最小值，当前值与最小值比较，如果小于最小值说明为错误元素应该包括进结果区间，所以右端点应该是i。逆序遍历寻找左端点，记录最大值，当前值与最大值比较， 如果大于最大值说明为错误元素应该包括进结果区间，所以左端点应该是i。

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        # 正序遍历寻找右端点
        max_num = nums[0]
        right = 0
        for i in range(0, len(nums)):
            # 说明当前num[i]小于前面的最大值为错误的元素，nums[i]应该包括到结果区间，右端点应该是i
            if nums[i] < max_num:
                right = i
            max_num = max(nums[i], max_num)

        # 逆序遍历寻找左端点
        min_num = nums[-1]
        left = len(nums) 
        for i in range(len(nums) - 1, -1, -1):
            # 说明当前num[i]大于后面的最小值为错误的元素，nums[i]应该包括到结果区间，左端点应该是i
            if nums[i] > min_num:
                left = i
            min_num = min(nums[i], min_num)
            
        return max(right - left + 1, 0)
```

#### 187.最长连续序列（128-Meidum）

##### Key：哈希表（用空间换时间）

题意：给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

分析：解题思路：要找到nums中有哪些元素能够当做连续序列的左边界。如果a为一个连续序列的左边界，则a-1就不可能存在于数组中。所以若一个元素值a满足：a-1不在nums数组中，则该元素值a就可以当做连续序列的左边界。利用set存储nums中的元素，当a可以充当左边界时，判断a+1、a+2、a+3……等是否存在数组中，并记录长度。每次记录最大的连续序列的长度，返回即可。

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # 哈希表：每次寻找能够充当左边界的元素值
        # 找到 nums 中有哪些元素能够当做连续序列的左边界
        if not nums:return 0
        nums = set(nums)
        max_length = 0
        for num in nums:
            # 如果num-1不在数组中，说明num可以充当连续序列的左边界，当前长度为1
            if num - 1 not in nums:
                current_num = num
                temp_length = 1
                # num是左边界，继续判断num+1,num+2,num+3等是否存在于集合中，并记录当前长度
                while current_num + 1 in nums:
                    current_num += 1
                    temp_length += 1
                # 每次记录最大的连续序列的长度
                max_length = max(temp_length, max_length)
        return max_length
```

#### 188.对称二叉树(101-Easy)

题意：给定一个二叉树，检查它是否是镜像对称的。

分析：可以递归也可以迭代，但是看了下还是迭代的速度和空间都更快一些。递归可以去代码文件看，迭代方法层序遍历，next_queue存储下一层结点，layer存储当前层结点的值，检查每一层是不是回文数组。

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:        
        # 迭代：层序遍历
        # next_queue存储下一层结点，layer存储当前层结点的值，检查每一层是不是回文数组
        queue = [root]
        while(queue):
            next_queue = list()
            layer = list()
            for node in queue:
                if not node:
                    layer.append(None)
                    continue
                next_queue.append(node.left)
                next_queue.append(node.right)       
                layer.append(node.val)
            if layer != layer[::-1]:
                return False
            queue = next_queue  
        return True
```

#### 189.变位词组(面试题10.02-Medium)

题意：编写一种方法，对字符串数组进行排序，将所有变位词组合在一起。变位词是指字母相同，但排列不同的字符串。

分析：变位词在排序之后是相同的，所以将排序后的作为key，原一些乱序的变位词作为value组合成键值对存入哈希表中，最后结果输出哈希表的所有值即可，注意存入的时候要以列表的形式，如果key存在就将当前append到值中。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 建立映射关系【排序后的字符串作为键，原字母相同排列不同的字符串作为值】
        hashtable = dict()
        for str in strs:
            # sort完的str是list，需要转换成字符串形式作为key值存储到哈希表中
            str_key = ''.join(sorted(str))
            # 将排序后的str作为key，原str作为value存储，注意是[str]
            if str_key not in hashtable:
                hashtable[str_key] = [str]
            else:
            # 当前key存在的话，直接存储到对应key的value值下面
                hashtable[str_key].append(str)
        # 遍历哈希表中所有值即为结果
        return [value for value in hashtable.values()]
        
```

#### 190.验证二叉搜索树（98-Meduim）

题意：给定一个二叉树，判断其是否是一个有效的二叉搜索树。假设一个二叉搜索树具有如下特征：节点的左子树只包含小于当前节点的数。节点的右子树只包含大于当前节点的数。所有左子树和右子树自身必须也是二叉搜索树。

分析：验证二叉搜索树即中序遍历得到升序序列即可，所以先对二叉树进行中序遍历然后判断是否为升序，检查升序需要注意两点：检查list里的数有没有重复元素，以及是否按从小到大排列。

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        # 迭代法构建中序遍历，res存储中序遍历值
        stack = []
        res = []
        cur = root
        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        # 检查list里的数有没有重复元素，以及是否按从小到大排列
        return res == sorted(res) and len(set(res)) == len(res)
```

#### 191.最高频元素的频数(1838-Meidum)

##### Key:滑动窗口

题意：元素的频数是该元素在一个数组中出现的次数。给你一个整数数组 nums 和一个整数 k 。在一步操作中，你可以选择 nums 的一个下标，并将该下标对应元素的值增加 1 。执行最多 k 次操作后，返回数组中最高频元素的 最大可能频数。

分析：排序 + 滑动窗口，右指针向前搜索，计算每次右移一位需要增加的数，*total表示每一位都向当前right所指的值对齐的话需要走多少步长。例如[1,2,4]，1向2对齐需要1步，1和2向4对齐：total已经等于1，这个1相当于1已经变成2的步长，那么2和2变成4的步长就是 2 \* (4-2) = 4 再加1=5，表示1和2向4对齐所需的步。当total步长大于k的时候需要移动左指针缩小窗口，left所指元素到right所指元素需要的步长早total中减掉还回去，同时窗口右移。实时记录窗口内元素的个数，因为窗口内的元素均是可以在规定步长内达到right所指元素的，所以满足条件。

```python
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        # 排序 + 滑窗
        nums.sort()
        N = len(nums)
        left,right = 0, 1
        # 最高频元素必定可以是数组中已有的某一个元素，所以初始化为 1
        res = 1
        total = 0
        # 右指针向前搜索
        while right < N:
            # 每次右移一位需要增加的数，total表示每一位都向当前right所指的值对齐的话需要走多少步长
            # 1,2,4,1向2对齐需要1步，1和2向4对齐：total已经等于1，这个1相当于1已经变成2的步长，那么2和2变成4的步长就是 2 * (4-2) = 4 再加1=5，表示1和2向4对齐所需的步长
            total += (right-left)*(nums[right] - nums[right-1])
            # 如果大于k，窗口右移，减去最左边数需要对齐到当前right需要的步长数【还回去】
            while total > k:
                total -= nums[right] - nums[left]
                left += 1
            # 实时记录窗口内的个数（窗口内都是可以在k之内对齐到当前right的）
            res = max(res, right - left + 1)
            right += 1
        return res
```

#### 192.数组中最大数对和的最小值(1877-Meidum)

##### Key:贪心算法

题意：一个数对 (a,b) 的 数对和 等于 a + b 。最大数对和 是一个数对数组中最大的 数对和 。比方说，如果我们有数对 (1,5) ，(2,3) 和 (4,4)，最大数对和 为 max(1+5, 2+3, 4+4) = max(6, 5, 8) = 8 。给你一个长度为 偶数 n 的数组 nums ，请你将 nums 中的元素分成 n / 2 个数对，使得：nums 中每个元素 恰好 在 一个 数对中，且最大数对和 的值 最小 。请你在最优数对划分的方案下，返回最小的 最大数对和 。

分析：尽量让较小数和较大数组成数对，先把数组排序，然后前后两两组合，提取数对中的最大值。

```python
class Solution:
    def minPairSum(self, nums: List[int]) -> int:
        # 贪心
        # 尽量让较小数和较大数组成数对
        # 对原数组 nums 进行排序，然后从一头一尾开始往中间组「数对」，取所有数对中的最大值
        nums.sort()
        left= 0
        right = len(nums) - 1
        res = 0
        for i in range(len(nums)):
            if left > right: break
            res = max(res, nums[left] + nums[right])
            left += 1
            right -= 1
        return res
```

#### 193.单词搜索(79-Meduim)

##### Key：深度优先搜索/回溯法

题意：给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

分析：深度优先遍历寻找四种可能的方向，回溯法的经典问题

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        #深搜解法
        if not board:
            return False
        #搜索可能的起始位置
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board,word,i,j):
                    return True
        return False

    # 深度搜索
    def dfs(self,board,word,i,j):
        # dfs终止条件
        if len(word) == 0:
            return True
        # 边界终止条件
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[0]:
            return False
        # 不能搜索到之前的位置，设该位置为None
        tmp,board[i][j] = board[i][j],None
        # 向上下左右四个方向搜索
        res = self.dfs(board,word[1:],i-1,j) or self.dfs(board,word[1:],i+1,j) or self.dfs(board,word[1:],i,j-1) or self.dfs(board,word[1:],i,j+1)
        board[i][j] = tmp
        return res
```

#### 194.表现良好的最长时间段(1124-Medium)

##### Key：哈希表/单调栈

题意：给你一份工作时间表 hours，上面记录着某一位员工每天的工作小时数。我们认为当员工一天中的工作小时数大于 8 小时的时候，那么这一天就是「劳累的一天」。所谓「表现良好的时间段」，意味在这段时间内，「劳累的天数」是严格 大于「不劳累的天数」。请你返回「表现良好时间段」的最大长度。

分析：这道题是曾经华为给我的面试题，太难了呀~，目前有三种方法：暴力、单调栈、哈希表。哈希表的题解：https://www.bilibili.com/video/BV1Wt411G7vN?from=search&seid=17906628306756620390

```python
class Solution:
    def longestWPI(self, hours: List[int]) -> int:
        # 哈希表：记录负数和第一次出现的位置，相同sum的最小索引
        # 判断往前少一步的前缀和是否存在，存在则计算一下距离，不用考虑多两步或者多三步，因为一定包含在一步内
        hashMap = dict()
        res = 0
        preSum = 0
        for i in range(len(hours)):
            preSum += 1 if hours[i] > 8 else -1
            # 如果当前preSum > 0，则[0-i]一定是满足条件的区间，记录长度为i + 1
            if preSum > 0:
                res = i + 1
            else:
            # 如果当前前缀和不存在字典中则记录当前的下标，即为当前preSum的最小索引；存在过就不放
                if preSum not in hashMap:
                    hashMap[preSum] = i
                # 判断-1是否存在，如果存在计算一下距离
                if preSum - 1 in hashMap:
                    res = max(res, i - hashMap.get(preSum - 1))
        return res
```

#### 195.复制带随机指针的链表(138-Medium)

题意：给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点。

分析：我们通过遍历旧节点的方式创建新节点，这里用回溯实现了节点的遍历。因为有random指针的存在，需要利用哈希表来存起来旧节点到新节点的映射，这样可以确保我们创建的节点不重复且唯一，以实现当同一个node被不同的节点node1的next和node2的random同时指向时，确保指向的是同一个node。

```python
class Solution:
    hashMap = dict()
    def copyRandomList(self, head: 'Node') -> 'Node':
        # 浅拷贝： 返回地址一样的链表。
        # 深拷贝： 返回地址不一样，但关系一致的链表
        # 回溯法 + 哈希表：回溯遍历结点，哈希表存储旧结点到新结点的映射
        if head == None:
            return None
        if head in self.hashMap:
            return self.hashMap.get(head)
        newHead = Node(x = head.val)
        self.hashMap[head] = newHead
        newHead.next = self.copyRandomList(head.next)
        newHead.random = self.copyRandomList(head.random)
        return newHead
```

#### 196.从相邻元素对还原数组(1743-Medium)

题意：存在一个由 n 个不同元素组成的整数数组 nums ，但你已经记不清具体内容。好在你还记得 nums 中的每一对相邻元素。给你一个二维整数数组 adjacentPairs ，大小为 n - 1 ，其中每个 adjacentPairs[i] = [ui, vi] 表示元素 ui 和 vi 在 nums 中相邻。题目数据保证所有由元素 nums[i] 和 nums[i+1] 组成的相邻元素对都存在于 adjacentPairs 中，存在形式可能是 [nums[i], nums[i+1]] ，也可能是 [nums[i+1], nums[i]] 。这些相邻元素对可以 按任意顺序 出现。返回 原始数组 nums 。如果存在多种解答，返回 其中任意一个 即可。

分析：hash表记录每个元素相邻的元素；找出只有一个邻居的元素使其作为起始元素（只有一个邻居的元素必定是首尾）；根据hash表长度确定元素的个数，并建立一个其个数个0的列表；将起始元素及其邻居加入列表，并将其邻居作为下一个操作对象；通过hash表将操作对象（即在中间有两个邻居的元素）的另一个另一个邻居随后加入，并将其作为下一个对象；重复4、5操作直至达到列表长度即可。

知识点：二维数组的遍历可以是` for i,j in adjacentPairs:`

```python
class Solution(object):
    def restoreArray(self, adjacentPairs):
        """
        :type adjacentPairs: List[List[int]]
        :rtype: List[int]
        """
        # 哈希表记录每个元素相邻的元素，依据哈希表的个数建立结果列表
        adjustNum = defaultdict(list)
        for i, j in adjacentPairs:
            adjustNum[i].append(j)
            adjustNum[j].append(i)
        res = [0] * len(adjustNum)

        # 遍历寻找哈希表中元素对应邻居只有一个，即为首位元素
        for key in adjustNum:
            if len(adjustNum[key]) == 1:
                res[0] = key
                break
        # 根据表中记录的邻居顺序填充，每一位让哈希表索引前一位的邻居，注意除去前面重复元素
        for i in range(1, len(res)):
            for j in adjustNum[res[i-1]]:
                if j != res[i-2]:
                    res[i] = j
        return res
```

#### 197.得到子序列的最少操作次数（1713-Hard）

##### Key:贪心+二分

题意：给你一个数组 target ，包含若干 互不相同 的整数，以及另一个整数数组 arr ，arr 可能 包含重复元素。每一次操作中，你可以在 arr 的任意位置插入任一整数。比方说，如果 arr = [1,4,1,2] ，那么你可以在中间添加 3 得到 [1,4,3,1,2] 。你可以在数组最开始或最后面添加整数。请你返回 最少 操作次数，使得 target 成为 arr 的一个子序列。一个数组的 子序列 指的是删除原数组的某些元素（可能一个元素都不删除），同时不改变其余元素的相对顺序得到的数组。比方说，[2,7,4] 是 [4,2,3,7,2,1,4] 的子序列（加粗元素），但 [2,4,2] 不是子序列。

分析：本题要找最少操作次数，实际上就是找最长的公共子序列(这样需要的操作最少)，根据target中互不相同，我们知道每个数字对应的坐标唯一。于是最长公共子序列等价于arr用target的坐标转换后构成最长上升子序列。因为**不管怎么样，公共子序列在target中必然是从左到右的，那么他们的坐标自然是从小到大的**。返回值是target的长度减去最长上升子序列（即target和arr的最长公共子序列），直接求最长公共子序列O(n^2)会超时，所以要转换成最长上升子序列。求解最长上升子序列是leetcode300题，动态规划方法也会超时，所以要采用贪心加二分的方法来求解

```python
class Solution:
    def minOperations(self, target: List[int], arr: List[int]) -> int:
        # 求最长公共子序列（操作最少）等价于arr用target的坐标转换后构成最长的上升子序列
        # 转换为最长上升子序列问题：arr用target的坐标转换后构成最长的上升子序列
        # 求最长上升子序列用贪心+二分，dp超时，答案len(target) - len(最长上升子序列)
        
        # 存储target元素及其对应下标
        hashmap = {}
        for i in range(len(target)):
            hashmap[target[i]] = i 
        # 在set中查找比在list下查找要快，不变set的话就会超时（虽然说target中已经是不同元素）
        target = set(target)
        
        # 将arr与target共同元素在表中遍历找到在target中的下标存在nums列表中
        nums = []
        for item in arr:
            if item in target:
                nums.append(hashmap[item])
        # 如果不存在公共元素，则target中所有元素均要操作
        if not nums:
            return len(target)

        # 寻找nums列表中最长上升子序列
        d = [] # 存储nums中最长上升子序列
        for num in nums:
            # 返回新的元素放在相等元素前面的位置，即
            pos = bisect.bisect_left(d,num)
            # 如果num > d[-1]将num直接加入d的后面
            if pos == len(d):
                d.append(num)
            # num插入到：第一个比num小的数d[i]的后面
            else:
                d[pos] = num
        # 最少操作次数 = target的长度减去最长公共子序列的长度
        return len(target) - len(d)
```

#### 198.二叉树中第二小的节点(671-Easy)

Key：深度优先搜索

题意：给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。如果一个节点有两个子节点的话，那么该节点的值等于两个子节点中较小的一个。更正式地说，root.val = min(root.left.val, root.right.val) 总成立。给出这样的一个二叉树，你需要输出所有节点中的第二小的值。如果第二小的值不存在的话，输出 -1 。

分析：二叉树根节点x的值小于以x为根的子树中所有节点的值，遍历子树所有节点，寻找严格大于根节点的最小值，采用深度优先遍历所有节点，如果没有的话返回float('inf')无穷大，如果当前根节点大于此时的最小值，那么就返回当前根节点的值，再递归左右子树返回最小值。

```python
class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        # 二叉树根节点x的值小于以x为根的子树中所有节点的值
        # 深度优先遍历，寻找严格比根节点小的值
        res = self.dfs(root, root.val)
        return res if res != float('inf') else -1

    def dfs(self, root, min_):
        if not root:
            return float('inf')
        if root.val > min_:
            return root.val
        return min(self.dfs(root.left, min_), self.dfs(root.right, min_))
```

#### 200.二叉树中所有距离为K的结点(863-Medium)

题意：给定一个二叉树（具有根结点 root）， 一个目标结点 target 和一个整数值 K 。返回到目标结点 target 距离为 K 的所有结点的值的列表。答案可以以任何顺序返回。

分析：先将二叉树转换成图，利用深度优先搜索，然后利用广度优先搜索从目标结点开始寻找左右上三个方向结点计算距离，再由三个结点循环扩散开记录距离。

```python
class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        # DFS深度优先遍历将树转换成图，增加parent指针
        def dfs(root, parent = None):
            if root:
                root.parent = parent
                dfs(root.left, root)
                dfs(root.right, root)
        dfs(root)

        # BFS广度优先遍历从目标结点开始寻找左右上三个方向结点计算距离，再由三个结点循环扩散开记录距离
        q = [(target, 0)]
        seen = {target}
        while q:
            # queue里面存储的元组对都是距离target有相同的距离（前一距离结点已被弹出）
            # 所以只需判断第一个是否满足，满足的话所有结点值全部输出
            if q[0][1] == k:
                return [node[0].val for node in q]
            # 弹出当前需遍历的结点和距离，以其为中心继续向左右上扩散计算
            node, distance = q.pop(0)
            for neighbor in (node.left, node.right, node.parent):
                if neighbor and neighbor not in seen:
                    seen.add(neighbor)
                    q.append((neighbor, distance + 1))
        return []
            
```

#### 201.二叉树寻路(1104-Medium)

题意:在一棵无限的二叉树上，每个节点都有两个子节点，树中的节点 逐行 依次按 “之” 字形进行标记。如下图所示，在奇数行（即，第一行、第三行、第五行……）中，按从左到右的顺序进行标记；而偶数行（即，第二行、第四行、第六行……）中，按从右到左的顺序进行标记。给你树上某一个节点的标号 label，请你返回从根节点到该标号为 label 节点的路径，该路径是由途经的节点标号所组成的。<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/06/28/tree.png" alt="img" style="zoom: 33%;" />

分析：首先如果先根据顺序完全二叉树的排列方式求出已知当前label为目标结点，求父节点的到目标结点的路径，从当前label开始，求其父节点`label//2`将路径添加到res中，最后添加根节点将其翻转。其次根据题意所说的逆序标记规则，当前label所在行不用翻转，那么label的上一行需要翻转，从刚刚路径数组的倒数第二行开始，每隔一个翻转一个值，此时的路径数组是由从根节点到目标结点经过每行的结点值。翻转公式为`本行最大值 - （当前值 - 本行开始值）`，对于顺序完全二叉树来说，每一行的起始为2的n次方，n从0开始，行末尾为2的(n+1)次方-1，因为下一行初始是2的(n+1)次方，所以依据翻转公式来对路径数组部分值进行翻转后即为所求答案。两步：一求正常路径二根据路径逆数翻转对应值。

```python
class Solution:
    def pathInZigZagTree(self, label: int) -> List[int]:
        # 顺序完全二叉树已知目标结点label求父节点（node//2）路径
        res = []
        while label != 1:
            res.append(label)
            label //= 2
        # 最后添加根节点且调整顺序为正序
        res.append(1)
        res.reverse()

        # 对于当前路径：从倒数第二个开始，每隔一个，找出取反相对应的值
        # 取反公式：本行最大值 - （当前值 - 本行开始值）
        # 完全二叉树每一层由 2**n (n从0开始)开始，最大值为 2**(n + 1)[下一层的起始] - 1
        for i in range(len(res) - 2, -1 ,-2):
            origin = res[i]
            start = 2 ** i
            end = 2 ** (i + 1) - 1
            new = end - (origin - start)
            res[i] = new
        return res

```

#### 202.Excel表列序号（171-Easy)

题意：给你一个字符串 `columnTitle` ，表示 Excel 表格中的列名称。返回该列名称对应的列序号。 A -> 1;B -> 2;C -> 3...;Z -> 26;AA -> 27;AB -> 28 

分析：找规律：单字母为ord('x') - ord('A') + 1，双字母先按照单字母求解第一个字母，结果为起始倍数*26加上第二个单字母。

```python
class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        # 找规律
        # 单字母为ord('x') - ord('A') + 1
        # 双字母先按照单字母求解第一个字母，结果为起始倍数*26加上第二个单字母
        ans = 0
        for item in columnTitle:
            ans = ans * 26 + ord(item) - ord('A') + 1
        return ans
```

#### 203.Excel表列名称(168-Easy)

题意：给你一个整数 `columnNumber` ，返回它在 Excel 表中相对应的列名称。

分析：上面一题的逆序，前面要加一注意这里要减一，注意不要用+=，会影响输出字母的顺序

```python
class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        res = ''
        while columnNumber:
            # A对应的序号是1不是0，所以列号要减1
            columnNumber -= 1
            # 这里不能用+=要用+，不然28应该是AB会输出BA，先计算出来的字符在后面
            res = chr(columnNumber % 26 + 65) + res
            columnNumber //= 26
        return res
```

#### 204.二叉树的垂序遍历(987-Hard)

题意:给你二叉树的根结点 root ，请你设计算法计算二叉树的 垂序遍历 序列。对位于 (row, col) 的每个结点而言，其左右子结点分别位于 (row + 1, col - 1) 和 (row + 1, col + 1) 。树的根结点位于 (0, 0) 。二叉树的 垂序遍历 从最左边的列开始直到最右边的列结束，按列索引每一列上的所有结点，形成一个按出现位置从上到下排序的有序列表。如果同行同列上有多个结点，则按结点的值从小到大进行排序。返回二叉树的 垂序遍历 序列。

分析：先利用dfs遍历树存储坐标到哈希表中，纵坐标为键，横坐标和结点值组成元组作为值，然后再对哈希表排序输出：先按照key排序，每个key下：按照value的第一维横坐标进行升序直接sorted，输出value的第二维结点值。

```python
class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        # 纵坐标为键，横坐标和结点值组成元组作为值，用哈希表记录坐标位置，用深度优先搜索进行遍历
        hashmap = defaultdict(list)
        def dfs(node, x, y):
            if not node:
                return 
            hashmap[y].append((x, node.val))
            dfs(node.left, x + 1, y - 1)
            dfs(node.right, x + 1, y + 1)
        dfs(root, 0, 0)

        # 先按照key排序，每个key下：按照value的第一维横坐标进行升序直接sorted，输出value的第二维结点值
        res = []
        for i in sorted(hashmap.keys()):
            temp = []
            for _,val in sorted(hashmap[i]):
                temp.append(val)
            res.append(temp)
        return res
        # 将上诉浓缩成return [[val for _, val in sorted(hashmap[x])] for x in sorted(hashmap)]
```

#### 205.矩阵中战斗力最弱的K行（1337-Easy）

题意：给你一个大小为 m * n 的矩阵 mat，矩阵由若干军人和平民组成，分别用 1 和 0 表示。请你返回矩阵中战斗力最弱的 k 行的索引，按从最弱到最强排序。如果第 i 行的军人数量少于第 j 行，或者两行军人数量相同但 i 小于 j，那么我们认为第 i 行的战斗力比第 j 行弱。军人 总是 排在一行中的靠前位置，也就是说 1 总是出现在 0 之前。

分析：直接朴素的解法先遍历后排序，遍历时候统计组号和对应1的个数

```python
class Solution:
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        # 先遍历后排序
        # 将每一组序号和该组1的个数组成元组形式存到res列表
        # 按照第二维升序排序，然后遍历获取前k个第一维的值
        res = []
        for i, row in enumerate(mat):
            res.append((i, row.count(1)))
        res.sort(key = lambda x:x[1])
        return [index for index, val in res[:k]]
```

#### 206.第N个泰波那契数列(1137-Easy)

题意：泰波那契序列 Tn 定义如下： T0 = 0, T1 = 1, T2 = 1, 且在 n >= 0 的条件下 Tn+3 = Tn + Tn+1 + Tn+2给你整数 n，请返回第 n 个泰波那契数 Tn 的值。

分析：看评论说递归会超时，所以就得这样写啦

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        # 前三个数的处理
        if n == 0:  return 0
        if n <= 2:  return 1
        # 初始化
        a, b, c = 0, 1, 1
        res = 0
        # 更新abc
        for i in range(3,n+1):
            res = a + b + c
            a, b, c = b, c, res
        return res
```

#### 207.等差数列划分(413-Meidum)

##### Key:动态规划

题意：如果一个数列 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该数列为等差数列。例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的子数组个数。子数组是数组中的一个连续序列。

分析：「自底向上」的思路，定义 dp[i] 是以 nums[i] 为终点的等差数列的个数，判断新增加的nums[i]可以和前面构成等差数列，如果可以的话记录到dp[i]中，状态转移方程为`dp[i] = dp[i-1] + 1`，最后统计dp数组中等差数列的个数。

```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # dp[i]表示以nums[i]结尾的等差数列数组的个数
        N = len(nums)
        dp = [0] * N
        for i in range(2, N):
            # 新增加的nums[i]可以和前面构成等差数列，dp[i] = dp[i-1] + 1
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                dp[i] = dp[i-1] + 1
        # 数组中的等差数列的数目
        return sum(dp)

```

#### 208.等差数列划分II-子序列(446-Hard)

##### Key:二维动态规划 + 哈希表

题意：给你一个整数数组 nums ，返回 nums 中所有 等差子序列 的数目。如果一个序列中 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该序列为等差序列。例如，[1, 3, 5, 7, 9]、[7, 7, 7, 7] 和 [3, -1, -5, -9] 都是等差序列。再例如，[1, 1, 2, 5, 7] 不是等差序列。数组中的子序列是从数组中删除一些元素（也可能不删除）得到的一个序列。例如，[2,5,10] 是 [1,2,1,2,4,1,5,10] 的一个子序列。题目数据保证答案是一个 32-bit 整数。

分析：这道题和上一道题的差异是这道题目可以去掉若干个元素，计算以 nums[i]、结尾的等差数列, 只要知道前面的「公差」和 nums[i]、本身的值，就能知道能不能跟前面组成等差数列。昨天题目的延伸，求所有子数组的数目，就是先计算每个nums[i]可以组成多少个，整体再求和。如图所示。

<img src="https://pic.leetcode-cn.com/1628652009-wwTndU-446.png" alt="446.png" style="zoom: 50%;" />

因为以nums[i]结尾的等差数列可能存在多个公差的等差数列，所以在第二维度的动态规划中使用哈希表来存储不同公差对应等差数列的个数。状态定义：`dp[i][j] 代表以 nums[i] 为结尾数字，能够组成公差为 j 的等差数列的个数。`，第一维表示以nums[i]结尾，第二维表示在第一维且公差为j的前提下对应的等差数列的个数，用字典存储键为公差值为数列个数。状态转移方程为：`dp[i][d] += dp[j][d] + 1`，累加`dp[j][d]`作为结果返回。

```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # dp[i][j] 代表以 nums[i] 为结尾数字，能够组成公差为 j 的等差数列的个数。
        # 因为以nums[i]结尾的等差数列可能存在多个公差的等差数列
        # 所以第二维度用哈希表存储不同公差对应等差数列的个数
        N = len(nums)
        dp = [defaultdict(int) for _ in range(N+1)]
        res = 0
        for i in range(N):
            for j in range(i):
                d = nums[i] - nums[j]
                # 更新 i 下面的 hashmap
                dp[i][d] += dp[j][d] + 1
                # 前一个结果作为数组下标 i 可以组成的等差数列数组个数
                res += dp[j][d]
        return res
```

#### 209.合并区间（56-Meidum）

题意：以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

分析：按照区间的左端点排序，那么在排序完的列表可以合并的区间一定是连续的。用数组res存储最后的答案，首先按照左端点升序排序，将第一个区间加入res数组中，顺序依次考虑后面的区间，如果数组中最后一个区间的右端点>=当前区间的左端点，那么可以合并，用当前区间的右端点更新数组最后一个区间的右端点取其两者较大值。没有合并就将原区间加入到结果数组中。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 按照左边界排序
        intervals.sort()
        # 初始放入第一个区间
        res = [intervals[0]]
        for i in range(1, len(intervals)):
            # 判断res里区间的右边界 >= 当前区间的左边界时可以合并
            if res[-1][1] >= intervals[i][0]:
                # 每次合并都取最大的右边界，左边界不动，res内始终是合并好的区间
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                # 没有合并加入原区间到结果数组中
                res.append(intervals[i])
        return res
```

#### 210.下一个排列(31-Medium)

题意：实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列（即，组合出下一个更大的整数）。如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。必须原地修改，只允许使用额外常数空间。

分析：原题的意思为需要找到一个比当前排列大的下一个排列，但是变大的幅度要尽可能小，也就是比123大的下一个排列是132，思路就是我们要寻找尽量靠右的【较小数】和右边尽可能小的【较大数】进行交换，交换完成后较大数的右边需要升序排列才能保证变大的幅度尽可能小。所以分三步：一，逆序遍历寻找第一处断崖的地方，即递减的地方，二，将断崖点右侧全部逆序进行升序排列(按照第一步来说原来是降序)，三，如果下一个排列存在的话，则交换断崖处i-1的较小数和右边升序序列中的较大数，完成！我严重且深刻的怀疑我写的这些我以后真的能看懂吗哈哈哈哈

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 题意：找到一个大于当前序列的新序列，且变大的幅度尽可能小
    # 思路：寻找尽量靠右的较小数和右边尽可能小的一个较大数交换，交换完成后较大数右边需升序排列
        # 逆序寻找尽量靠右的较小数，从右向左一路上升寻找第一处断崖点i-1
        N = len(nums)
        if N <= 1:
            return 
        i = N - 1
        while i > 0 and nums[i-1] >= nums[i]:
            i -= 1
        
        # 找到断崖处i-1，将右边全部升序排列
        l = i
        r = N - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1

        # 如果下一个排列存在的话
        # 按照升序序列寻找较大数与断崖处的较小数进行交换
        if i != 0:
            for j in range(i,N):
                if nums[i-1] < nums[j]:
                    nums[i-1], nums[j] = nums[j], nums[i-1]
                    break
```

#### 211.在排序数组中查找元素的第一个和最后一个位置（34-Meidum）

##### Key:二分查找

题意：给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。如果数组中不存在目标值 target，返回 [-1, -1]。

进阶：你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？

分析：利用到数组升序排列的条件，采用二分查找，即为寻找 开始位置[第一个等于target的位置] 和 结束位置[第一个大于target的位置-1] ，手写二分查找bisect_left模块。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 利用升序条件，使用二分查找
        # 即为寻找 [第一个等于target的位置] 和 [第一个大于target的位置-1] 
        
        # 二分查找bisect_left模块（查找元素所在位置并返回，相同元素返回应插入左边位置）
        def bisect_left(target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        # 获取target位置和target下一个数字的应插入位置-1即为开始和结束位置
        begin = bisect_left(target)
        end = bisect_left(target + 1) - 1

        # 不存在target
        if begin == len(nums) or nums[begin] != target:
            return [-1, -1]
        
        return[begin, end]

```

#### 212.最长回文子序列(516-Medium)

##### Key：动态规划

题意：给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

分析：子串要求是连续的，子序列可以不连续，状态定义：`dp[i][j]：字符串s在[i, j]范围内最长的回文子序列的长度为dp[i][j]`，在判断回文子串的题目中，关键逻辑就是看s[i]与s[j]是否相同，如果相同则长度+2，如果s[i]与s[j]不相同，说明s[i]和s[j]的同时加入 并不能增加[i,j]区间回文子串的长度，那么分别加入s[i]、s[j]看看哪一个可以组成最长的回文子序列。根据递推公式推导遍历顺序，注意i是从下到上。一篇很好很形象的题解：https://leetcode-cn.com/problems/longest-palindromic-subsequence/solution/dai-ma-sui-xiang-lu-dai-ni-xue-tou-dpzi-dv83q/

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # dp[i][j]表示s[i:j]最长回文子序列的长度
        N = len(s)
        dp = [[0] * N for _ in range(N)]
        # 每个字符都是单独的回文子序列
        for i in range(N):
            dp[i][i] = 1
        for i in range(N-1, -1, -1):
            for j in range(i+1, N):
                # 判断s[i]与s[j]是否相同，相同则长度加2，不同则判断加入其中之一哪个形成的回文子序列最长
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return dp[0][-1]
```

#### 213.反转字符串中的元音字母(345-Easy)

##### Key：双指针

题意：编写一个函数，以字符串作为输入，反转该字符串中的元音字母。

分析：左右指针从前和后两个方向向着中间遍历，找到元音字母就交换

```python
class Solution:
    def reverseVowels(self, s: str) -> str:
        # 双指针，注意字符串先转列表才能做交换
        s = list(s)
        left, right = 0 , len(s) - 1
        vowel_char = ['a', 'i', 'e', 'o', 'u', 'A', 'I', 'E', 'O', 'U']
        # 判断左右指针所到位置是否为元音字母，如果是就交换元音
        while left < right:
            while left < right and s[left] not in vowel_char:
                left += 1
            while left < right and s[right] not in vowel_char:
                right -= 1
            s[left], s[right] = s[right], s[left]
            # 交换完之后左右指针 
            left += 1
            right -= 1
        # 列表再转字符串
        return ''.join(s)
```

#### 214.和为K的子数组(560-Medium)

题意：给定一个整数数组和一个整数 **k，**你需要找到该数组中和为 **k** 的连续的子数组的个数。

分析：将连续子数组之和表示为前缀和之差，利用哈希表存储前缀和作为键，出现的次数作为值，类似于两数之和的解法，判断pre - k是否存在于字典中，存在就记录下次数。如果用collections.defaultdict(int)就不需要判断是否存在设置为0，因为好像初始化默认就都是0.这里要注意一下`hashTable[0] = 1`这个铺底，当数组中有0的时候，不设置这个就会少计数一次：例如{1,1,0} k = 2的时候应该是2，1+1和1+1+0。

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # 将连续子数组之和表示为前缀和之差pre[i]-pre[j] (j<i)
        # 前缀和 + 哈希表存储前缀和出现的次数
        pre, res, hashTable = 0, 0, {}
        # 当数组中有0的时候，不设置这个就会少计数一次：例如{1,1,0} k = 2的时候应该是2,1+1和1+1+0
        hashTable[0] = 1
        for i in range(len(nums)):
            pre += nums[i]
            if pre - k in hashTable:
                res += hashTable[pre - k]
            if pre not in hashTable:
                hashTable[pre] = 0
            hashTable[pre] += 1
        return res
```

#### 215.找到字符串中所有字母异位词(438-Medium)

##### Key:滑动窗口

题意：给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。异位词 指字母相同，但排列不同的字符串。

分析：因为字符串中的字符全是小写字母，可以用长度为26的数组记录字母出现的次数。设m = len(s), n = len(p)。记录p字符串的字母频次p_cnt，和s字符串前n个字母频次s_cnt
若p_cnt和s_cnt相等，则找到第一个异位词索引 0继续遍历s字符串索引为[n, m)的字母，在s_cnt中每次增加一个新字母，去除一个旧字母。判断p_cnt和s_cnt是否相等，相等则在返回值res中新增异位词索引 i - n + 1
链接：https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/solution/438-zhao-dao-zi-fu-chuan-zhong-suo-you-z-nx6b/

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 滑动窗口 + 数组
        
        m, n, res = len(s), len(p), []
        if m < n:return res
        # 用数组记录字母出现的次数
        books_ord = [0] * 26
        bookp_ord = [0] * 26
        # 记录n个字母频次
        for i in range(n):
            books_ord[ord(s[i]) - ord('a')] += 1
            bookp_ord[ord(p[i]) - ord('a')] += 1
        
        # 首个窗口判断
        if books_ord == bookp_ord:
            res.append(0)
        # 剩余窗口判断
        for i in range(n,m):
            # 增加当前遍历字符，去除左边窗口旧字符
            books_ord[ord(s[i])-ord('a')] += 1
            books_ord[ord(s[i-n])-ord('a')] -= 1
            if books_ord == bookp_ord:
                res.append(i - n + 1)
        return res

        
```

#### 216.岛屿数量(200-Meidum)

##### Key:深度优先搜索

题意:给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。此外，你可以假设该网格的四条边均被水包围。

分析：扫描整个二维网格，以1为起点进行深度优先搜索，在深度优先搜索的过程中，每个搜索到的1都会被重新标记为0，然后在该点的四个方向进行搜索，查到1的话继续dfs，最终岛屿的数量就是我们进行深度优先搜索的次数。

```python
class Solution:
    # 深度优先搜索
    # 以位置1为起点进行深度优先搜索，搜索到的1标记为0
    def dfs(self, grid, r, c):
        grid[r][c] = 0
        nr, nc = len(grid), len(grid[0])
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs(grid, x, y)

    # 最终岛屿的数量就是深度优先搜索的次数
    def numIslands(self, grid: List[List[str]]) -> int:
        nr, nc = len(grid), len(grid[0])
        if nr == 0:
            return 0
        num_island = 0
        for i in range(nr):
            for j in range(nc):
                if grid[i][j] == "1":
                    num_island += 1
                    self.dfs(grid, i, j)
        return num_island
```

#### 217.跳跃游戏(55-Medium)

##### Key:贪心

题意：给定一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标。

分析：依次遍历数组中的每一个位置，并实时维护**最远可以到达的位置**。对于当前遍历到的位置 x，如果它在最远可以到达的位置的范围内，那么我们就可以从起点通过若干次跳跃到达该位置，因此我们可以用 x+nums[x] (当前位置加上最远可以跳的步数即为最远可到达的位置）更新**最远可以到达的位置**。在遍历的过程中，如果最远可以到达的位置大于等于数组中的最后一个位置，那就说明最后一个位置可达，我们就可以直接返回 True 作为答案。反之，如果在遍历结束后，最后一个位置仍然不可达，我们就返回 False 作为答案。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
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
```

#### 218-最小K个数（面试题17.14）

##### Key：堆排序

题意：设计一个算法，找出数组中最小的k个数。以任意顺序返回这k个数均可。

分析：用一个大根堆实时维护数组的前 k 小值。首先将前 k 个数插入大根堆中，随后从第 k+1 个数开始遍历，如果当前遍历到的数比大根堆的堆顶的数要小，就把堆顶的数弹出，再插入当前遍历到的数。最后将大根堆里的数存入数组返回即可。python默认是小根堆，所以要存入相反数

```python
class Solution:
    def smallestK(self, arr: List[int], k: int) -> List[int]:
        # 用大根堆实时维护数组前k小值，由于python自带小根堆所以存入相反数
        if k == 0:
            return list()
        import heapq
        hp = [-x for x in arr[:k]]
        heapq.heapify(hp)
        for i in range(k, len(arr)):
            if -hp[0] > arr[i]:
                # 弹出最小元素并将-arr[i]压入堆中，相当于pop再push
                heapreplace(hp, -arr[i])
        return [-x for x in hp]
```

#### 219-字符串解码(394-Medium)

题意：给定一个经过编码的字符串，返回它解码后的字符串。编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

分析：本题核心思路是在栈里面每次存储两个信息, (左括号前的字符串, 左括号前的数字), 比如abc3[def], 当遇到第一个左括号的时候，压入栈中的是("abc", 3), 然后遍历括号里面的字符串def, 当遇到右括号的时候, 从栈里面弹出一个元素(s1, n1), 得到新的字符串为s1+n1*"def", 也就是abcdefdefdef。对于括号里面嵌套的情况也是同样处理方式。凡是遇到左括号就进行压栈处理，遇到右括号就弹出栈，栈中记录的元素很重要。

```python
class Solution:
    def decodeString(self, s: str) -> str:
        # 栈内存储信息对（左括号前的字符串，左括号前的数字）
        stack = []
        num = 0
        res = ''
        for c in s:
            # 遇到数字记录下来，
            if c.isdigit():
                # 如果是两位数比如32，那么num先记录3，再记录3*10+2
                num = num * 10 + int(c) 
            # 遇到左括号压入栈，清零开始遍历括号内的字符串
            elif c == '[':
                stack.append((res,num))
                res, num = '', 0
            # 遇到右括号，栈内弹出元素计算当前新的字符串
            elif c == ']':
                top = stack.pop()
                res = top[0] + res * top[1]
            # 遇到字符直接记录
            else:
                res += c
        return res
```

#### 220.二分查找(704-Easy)

题意：给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

分析:背住背住，左闭右闭

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 二分查找：左闭右闭
        left, right = 0, len(nums) - 1
        while left <= right:
            middle = (left + right) // 2
            if nums[middle] < target:
                left = middle + 1
            elif nums[middle] > target:
                right = middle - 1
            else:
                return middle
        return -1
```

#### 221.分割平衡字符串（1221-Medium）

题意：在一个 平衡字符串 中，'L' 和 'R' 字符的数量是相同的。给你一个平衡字符串 s，请你将它分割成尽可能多的平衡字符串。注意：分割得到的每个字符串都必须是平衡字符串。返回可以通过分割得到的平衡字符串的 最大数量 。

分析：贪心：从前向后遍历，只要遇到平衡子串，计数就+1，遍历一遍即可。局部最优：从前向后遍历，只要遇到平衡子串就统计。全局最优：统计了最多的平衡子串。局部最优可以推出全局最优

```python
class Solution:
    def balancedStringSplit(self, s: str) -> int:
        # 贪心：从前向后遍历，只要遇到平衡子串，计数就+1，遍历一遍即可。
        cnt, res= 0, 0
        for c in s:
            if c == 'L':
                cnt += 1
            else:
                cnt -= 1
            if cnt == 0:
                res += 1
        return res
```

#### 222.IPO（502-Hard）

题意：帮助力扣设计完成最多 k 个不同项目后得到最大总资本的方式。给你 n 个项目。对于每个项目 i ，它都有一个纯利润 profits[i] ，和启动该项目需要的最小资本 capital[i] 。最初，你的资本为 w 。当你完成一个项目时，你将获得纯利润，且利润将被添加到你的总资本中。总而言之，从给定项目中选择 最多 k 个不同项目的列表，以 最大化最终资本 ，并输出最终可获得的最多资本。答案保证在 32 位有符号整数范围内。

分析：我们需要知道的信息是: 在需要的本金小于等于我们当前资本的项目中，利益最大的是哪个，所以我们将输入组合起来并按本金排序，这样在循环中，所有小于等于当前本金的都可以加入到大顶堆中(维护所有可选的利益的堆)，于是我们就可以选到当前利益最大的那个的了。一种特殊情况是，如果我们当前的资本已经不支持任何项目了，也就是我们的钱再也不可能发生变化了，只能结束了。注: 每次都选利益最大的那个是因为我们选项目的个数有限制，选的越大最终的答案才能越大。

```python
import heapq
class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        # 贪心+大顶堆：每次选取所需资本最小但利益最高的项目
        N = len(profits)
        # 将profits和capital组合起来，并按本金排序，这样保证我们总能选取所有小于等于当前资本的
        project = sorted(zip(profits, capital), key = lambda x:x[1])
        idx = 0
        cur = []
        while k:
            # 将所有需要的本金小于等于当前资本的项目加入最大堆(加入为-)
            while idx < N and project[idx][1] <= w:
                heapq.heappush(cur, -project[idx][0])
                idx += 1
            # 如果当前大根堆内存在可以完成的项目，则选取堆顶最高利润项目弹出，累加到资本中。
            if cur:
                w -= heapq.heappop(cur) 
            else:
                break
            k -= 1
        return w
```

#### 223.回旋镖的数量(447-Medium)

##### Key:哈希表

题意：给定平面上 n 对 互不相同 的点 points ，其中 points[i] = [xi, yi] 。回旋镖 是由点 (i, j, k) 表示的元组 ，其中 i 和 j 之间的距离和 i 和 k 之间的距离相等（需要考虑元组的顺序）。返回平面上所有回旋镖的数量。

分析：对于每个回旋镖三元组而言，本质上我们在统计给定 i 的情况下，与 i 距离相等的 (j, k)组合个数为多少。使用哈希表进行预处理，在统计以 i 为三元组第一位的回旋镖个数前，先计算出 i 和其余点的距离，并以 { 距离 : 个数 } 的形式进行存储，然后分别对所有的距离进行累加计数。在计算距离时为了避免使用 sqrt，我们直接使用 x^2 + y^2 来代指两点间的距离。**注意python中表示平方是********，对相同距离的个数累加计算三元组的个数时，j和k的顺序可以调换，N个数的排列组合有N*(N-1)种。

```python
class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        # 哈希表存储每个点作为i点时，其他点距离i点的(距离：个数)
        # 对每个点都要单独建立一个哈希表，计算其余各点到该点的距离，存储到表中
        res = 0
        for pi in points:
            hashTable = dict()
            for pj in points:
                if pi == pj:continue
                # 计算以pi为起点的距离，将根号改为平方
                dist = (pi[0] - pj[0])**2 + (pi[1] - pj[1])**2
                hashTable[dist] = hashTable.get(dist,0) + 1
            # 相同距离的个数可以组成的三元组有n * (n-1)个【排列组合】
            for value in hashTable.values():
                res += value * (value - 1)
        return res
```

#### 224.寻找峰值(162-Medium)

题意：峰值元素是指其值严格大于左右相邻值的元素。给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。你可以假设 nums[-1] = nums[n] = -∞ 。

分析：利用二分查找寻找爬坡方向，主要判断 mid 和 mid + 1 的高度谁高，如果 mid + 1 更大， 说明 mid 之后肯定还在爬升，mid + 1 之后有峰，如果 mid 更大， 说明 mid 之前有峰。条件退出的时候 l 和 r 相等， 而我们始终保持 [l, r] 内有峰。 所以，l或者r就是峰所在的位置

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        # 二分查找
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            # 峰值可能出现在mid右侧
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            # 峰值可能出现在mid以及mid向左处
            else:
                right = mid
        # 当left和right相等指向峰值的时候，返回left
        return left
```

#### 225.有效的数独(36-Meidum)

题意：请你判断一个 9x9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。数字 1-9 在每一行只能出现一次。数字 1-9 在每一列只能出现一次。数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）数独部分空格内已填入了数字，空白格用 '.' 表示。注意：一个有效的数独（部分已被填充）不一定是可解的。只需要根据以上规则，验证已经填入的数字是否有效即可。

分析：遍历棋盘记录行、列、方块中的数字，用集合存储每行、每列、每个box内部的数字，注意box和i，j下标序号的转换:pos = (i//3)*3 + j//3。遍历判断每位数字是否满足三个条件，满足的话集合加入，不满足任一条件直接返回False。

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # 遍历棋盘记录行、列、方块中的数字，用集合存储每行、每列、每个box内部的数字
        # 注意box和i，j下标序号的转换:pos = (i//3)*3 + j//3

        row = [set() for i in range(9)]
        column = [set() for i in range(9)]
        box = [set() for i in range(9)]

        for i in range(9):
            for j in range(9):
                item = board[i][j]
                pos = (i // 3) * 3 + j // 3
                if item != '.':
                # 判断该数字是否已经在该行该列以及该box块是否出现过，如果没有出现过则添加到set中
                    if item not in row[i] and item not in column[j] and item not in box[pos]:
                        row[i].add(item)
                        column[j].add(item)
                        box[pos].add(item)
                # 如果出现过就返回False
                    else:
                        return False
        return True
```

#### 226.只有两个键的键盘(650-Meidum)

题意：最初记事本上只有一个字符 'A' 。你每次可以对这个记事本进行两种操作：Copy All（复制全部）：复制这个记事本中的所有字符（不允许仅复制部分字符）。
Paste（粘贴）：粘贴 上一次 复制的字符。给你一个数字 n ，你需要使用最少的操作次数，在记事本上输出 恰好 n 个 'A' 。返回能够打印出 n 个 'A' 的最少操作次数。

分析：动态规划，N是质数的话就等于n，合数的话找最大的因子+复制粘贴这个因子所需次数

```python
class Solution:
    def minSteps(self, n: int) -> int:
        # 动态规划：n是质数结果还是n只能一个一个复制，n是合数结果是因数所需最小操作次数+复制粘贴因数次数

        if n == 1:return 0
        # dp[i]表示打印i个‘A’的最少操作次数
        dp = [0] * (n+1)

        for i in range(2, n+1):
            dp[i] = i
            # i是合数的情况下寻找i的最大因数更新dp[i]
            for j in range(int(i/2), 1, -1):
                # 如果j是i的因数，i个可以由j个复制粘贴得到，复制粘贴次数为i/j
                if i % j == 0:
                    dp[i] = dp[j] + int(i/j)
                    break # 因为因数之前计算过，可以保证所需的步数最少，所以j从大到小可以break
        return dp[n]
```

#### 227.最长递增子序列的个数(673-Medium)

题意：给定一个未排序的整数数组，找到最长递增子序列的个数。

分析：动态规划！

（1）定义dp数组：dp[i]：i之前（包括i）最长递增子序列的长度为dp[i]；count[i]：以nums[i]为结尾的字符串，最长递增子序列的个数为count[i]

（2）确定递推公式：

​	在nums[i] > nums[j]前提下，如果在[0, i-1]的范围内，找到了j，使得dp[j] + 1 > dp[i]，说明找到了一个更长的递增子序列。

​	那么以j为结尾的子串的最长递增子序列的个数，就是最新的以i为结尾的子串的最长递增子序列的个数，即：count[i] = count[j]。

​	在nums[i] > nums[j]前提下，如果在[0, i-1]的范围内，找到了j，使得dp[j] + 1 == dp[i]，说明找到了两个相同长度的递增子序列。

​	那么以i为结尾的子串的最长递增子序列的个数 就应该加上以j为结尾的子串的最长递增子序列的个数，即：count[i] += count[j];

​	题目要求最长递增序列的长度的个数，所以利用maxCount记录最长长度。

​	最后还有再遍历一遍dp[i]，把最长递增序列长度对应的count[i]累计下来就是结果了。

（3）确定遍历顺序：dp[i]是由0到i-1各个位置的最长升序子序列推导而来，那么遍历i一定是从前向后遍历。j其实就是0到i-1，遍历i的循环里外层，遍历j则在内层。

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        # dp[i]以nums[i]结尾最长递增子序列的长度
        # count[i]以num[i]为结尾的最长递增子序列个数
        N = len(nums)
        if N <= 1: return N
        dp = [1 for i in range(N)]
        count = [1 for i in range(N)]
        
        maxCount = 0    # 记录最长递增子序列的长度
        for i in range(N):
            for j in range(i):
                if nums[i] > nums[j]:
                    # 找到一个更长的递增子序列
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    # 找到了两个相同长度的递增子序列
                    elif dp[j] + 1 == dp[i]:
                        count[i] += count[j]
                maxCount = max(maxCount,dp[i])
        
        # 再遍历一遍，把最长递增序列长度对应的count[i]累计下来
        result = 0
        for i in range(N):
            if dp[i] == maxCount:
                result += count[i]
        return result
                
```

#### 228.最后一个单词的长度(58-Easy)

题意：给你一个字符串 `s`，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中最后一个单词的长度。**单词** 是指仅由字母组成、不包含任何空格字符的最大子字符串。

分析：从后向前遍历，跳过末尾的空格确定最后一个单词的末尾起始点，然后向前遍历最后一个单词的长度，遇到空格跳出循环。

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        # 从后向前遍历
        start = len(s) - 1
        # 确定起始点，跳过末尾的大量空格
        while s[start] == " ":
            start -= 1
        # 从起始点向前遍历最后一个单词，遇到空格就跳出循环
        res = 0
        for i in range(start, -1, -1):
            if s[i] != " ":
                res += 1
            else:
                break
        return res
```

#### 229.分隔链表(725-Meidum)

题意：给你一个头结点为 head 的单链表和一个整数 k ，请你设计一个算法将链表分隔为 k 个连续的部分。每部分的长度应该尽可能的相等：任意两部分的长度差距不能超过 1 。这可能会导致有些部分为 null 。这 k 个部分应该按照在链表中出现的顺序排列，并且排在前面的部分的长度应该大于或等于排在后面的长度。返回一个由上述 k 部分组成的数组。

分析：主要步骤：（1）算出k个部分每个部分最少有几个节点；（2）算出对于全部的节点数，分成k部分多出来几个节点；（3）将多出来的节点平分在前面的部分中；（4）按照得到的每个部分的节点数量，将每个部分的开头节点纳入res，并切断后续链表

```python
class Solution:
    def splitListToParts(self, head: ListNode, k: int) -> List[ListNode]:
        # 计算k个部分每部分最少多少结点，分完k部分多出的结点平分到前面去
        # 按照每部分结点数量，将每部分的头结点纳入res，切断后续链表
        
        #1. 特殊情况判断
        if not head:
            return [None for _ in range(k)]
        
        #2. 计算链表长度 
        n = 0
        p = head
        while p:
            n += 1
            p = p.next
        
        #3. 计算k部分每部分最少多少结点，以及多余出来多少结点
        part_len = n // k
        remain_len = n % k
        # 列表存储k个值，每个值为每部分的长度
        part = [part_len] * k
        # 将多出来的结点均分在之前的结点上
        for i in range(remain_len):
            part[i] += 1
        
        #4. 开始遍历添加每一部分的起始结点到res中【注意head是ListNode型，append进去就是小List形式】
        res = [head]
        for i,num in enumerate(part):
            # 按照part内存储的长度遍历每一部分，head最后指向下一部分的起始结点，pre指向该遍历部分的最后结点
            for _ in range(num):
                pre = head
                head = head.next
            # 每次是在每一部分循环开始前添加该部分的头结点head，并切断上一部分最后结点的next。
            if i != k-1:
                pre.next = None
                res.append(head)
        return res
```

#### 230.3的幂(326-Easy)

题意：给定一个整数，写一个函数来判断它是否是 3 的幂次方。如果是，返回 true ；否则，返回 false 。整数 n 是 3 的幂次方需满足：存在整数 x 使得 n == 3x

分析：0和1特殊情况判断，不断除3然后判断mod3是不是0，如果不是返回False

```python
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n == 1:  return True
        if n == 0:  return False
        while n != 1:
            if n % 3 == 0:
                n //= 3
            else:
                return False
        return True
```

#### 231.两个字符串的删除操作(583-Medium)

题意：给定两个单词 word1 和 word2，找到使得 word1 和 word2 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

分析：最长公共子序列问题，先找到最长公共子序列的长度，然后用两个字符串分别减去最长公共子序列的长度剩下的字符就是需要操作的步数。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # 寻找两个字符串的最长公共子序列
        # 结尾：用两个字符串长度之和减去2倍的最长公共子序列就是需要的步数
        M = len(word1)
        N = len(word2)
        dp = [[0] * (N + 1) for _ in range(M+1)]
        for i in range(1, M+1):
            for j in range(1, N+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return M + N - 2 * dp[M][N]
```

#### 232.搜索插入位置(35-Easy)

题意：给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。请必须使用时间复杂度为 O(log n) 的算法。

分析：二分查找左闭右闭情况，如果在数组中存在目标值则直接返回middle，目标值不在数组中则返回left或者right+1。

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        # 二分查找左闭右闭
        left, right = 0, len(nums) - 1
        # (1)数组中找到目标值的情况直接返回下标
        while left <= right:
            middle = (left + right) // 2
            if target > nums[middle]:
                left = middle + 1
            elif target < nums[middle]:
                right = middle - 1
            else:
                return middle
        # 分别处理如下三种情况:
        # (2)目标值在数组所有元素之前 [0,-1] 
        # (3)目标值插入数组中的位置 [left, right] ，return right + 1 或者 return left 即可
        # (4)目标值在数组所有元素之后的情况 [left, right]，return right + 1 或者 return left 即可
        # left = right + 1
        return right + 1
```

#### 233.旅行终点站（1436-Easy)

题意：给你一份旅游线路图，该线路图中的旅行线路用数组 paths 表示，其中 paths[i] = [cityAi, cityBi] 表示该线路将会从 cityAi 直接前往 cityBi 。请你找出这次旅行的终点站，即没有任何可以通往其他城市的线路的城市。题目数据保证线路图会形成一条不存在循环的线路，因此恰有一个旅行终点站。 

分析：利用哈希表存储每条路径中的起点城市cityA，遍历路径中的cityB，如果在哈希表中找不到则说明为旅行终点站（不会再做任何一条路径的起点）

```python
class Solution:
    def destCity(self, paths: List[List[str]]) -> str:
        # 哈希表存储每条路径中的起点城市cityA
        cityA = {path[0] for path in paths}
        # 遍历路径中的cityB，如果在cityA的哈希表中不存在则说明该终点未做过其他路径的起点，即为旅行终点站
        for path in paths:
            if path[1] not in cityA:
                return path[1]
```

#### 234.数字转换为十六进制数(405-Easy)

题意：给定一个整数，编写一个算法将这个数转换为十六进制数。对于负整数，我们通常使用 补码运算 方法。

注意:

十六进制中所有字母(a-f)都必须是小写。
十六进制字符串中不能包含多余的前导零。如果要转化的数为0，那么以单个字符'0'来表示；对于其他情况，十六进制字符串中的第一个字符将不会是0字符。 
给定的数确保在32位有符号整数范围内。
不能使用任何由库提供的将数字直接转换或格式化为十六进制的方法。

分析：存在负数，将num看成32位二进制数，将其4个一组共八组转换为16进制。

```python
class Solution:
    def toHex(self, num: int) -> str:
        # 32位2进制数，转换成16进制 -> 4个一组，一共八组
        base = "0123456789abcdef"
        res = []
        for _ in range(8):
            res.append(num%16)
            num //= 16
            if not num:
                break
        return "".join(base[n] for n in res[::-1])
```

#### 235.第三大的数(414-Easy)

题意：给你一个非空数组，返回此数组中 **第三大的数** 。如果不存在，则返回数组中最大的数。

分析：三种方法，最直观的set去重+排序，其次想复习一下大顶堆），最后就是经典找第N大的数用一次遍历的解法。

```python
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        # set去重 + 排序
        temp = list(set(nums))
        temp.sort(reverse=True)
        if len(temp) < 3:
            return temp[0]
        else:
            return temp[2]
```

```python
import heapq
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        queue = []
        res = []
        heapq.heapify(queue)
        # 将数字相反数放入堆中，自动排序后堆顶元素即为负最大值
        for item in nums:
            heapq.heappush(queue, -item)
        # 弹出堆顶元素加入结果数组，注意判断是否已存在
        while queue and len(res) < 3:
            item = heapq.heappop(queue)
            if item not in res:
                res.append(item)
        # 返回结果数组中第三大的数，没有返回最大值
        return -res[-1] if len(res) == 3 else -res[0]
```

```python
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        # 利用三个变量存储最大值、次大值和第三大值
        a = b = c = float('-inf')
        for num in nums:
            if num > a:
                c, b, a = b, a, num
            elif num > b and num != a:
                c, b = b, num
            elif num > c and num != a and num != b:
                c = num
        return c if c != float('-inf') else max(nums)
```

#### 236.字符串中的单词数（434-Easy）

题意：统计字符串中的单词个数，这里的单词指的是连续的不是空格的字符。

分析：判断第一个字符不是空格的时候累加第一个单词，其他情况只要s[i]是空格，s[i-1]不是空格，res就加1。

```python
class Solution:
    def countSegments(self, s: str) -> int:
        # i = 0表示记录第一个字符不是空格的时候
        # 只要s[i]是空格，s[i-1]不是空格，res就加1
        res = 0
        for i in range(len(s)):
            if s[i] != ' ' and (i == 0 or s[i-1] == ' '):
                res += 1
        return res
```

#### 237.排列硬币(441-Easy)

题意：你总共有 n 枚硬币，并计划将它们按阶梯状排列。对于一个由 k 行组成的阶梯，其第 i 行必须正好有 i 枚硬币。阶梯的最后一行 可能 是不完整的。给你一个数字 n ，计算并返回可形成 完整阶梯行 的总行数。

分析：二分查找，第 0 ~ i 行总的硬币数目有 (i + 1) * i / 2，二分查找满足(i + 1) * i / 2  <= n的最大数字i

```python
class Solution:
    def arrangeCoins(self, n: int) -> int:
        # 等差数列计算公式可得：第 0 ~ i 行总的硬币数目有 (i + 1) * i / 2
        # 二分查找满足(i + 1) * i / 2  <= n的最大数字i

        # 二分查找双闭区间
        left, right = 0, n
        while left <= right:
            mid = (left + right) // 2
            if mid * (mid + 1)  <= 2 * n:
                left = mid + 1
            else:
                right = mid - 1
        # 最后left会在mid基础上多加一步，所以要减一
        return left - 1
```

#### 238.搜索二维矩阵II（240-Medium）

题意：编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：每行的元素从左到右升序排列。每列的元素从上到下升序排列。

分析：可以从左下角或者右上角开始分析

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 标志位：可以选择从左下角或者右上角开始遍历
        # 从左下角开始，左下角元素是这一行中最小，这一列中最大
        m = len(matrix)
        n = len(matrix[0])
        if m == 0 or n == 0:
            return False
        
        i = m - 1
        j = 0
        while i >= 0 and j < n:
            # 左下角等于目标则找到
            if matrix[i][j] == target:
                return True
            # 左下角元素小于目标，目标元素不可能在当前列，向右走规模可在去掉第一列的子矩阵中寻找
            elif matrix[i][j] < target:
                j = j + 1
            # 左下角元素大于目标，目标元素不可能在当前行，向上走规模可在去掉最后一行的子矩阵中寻找
            else:
                i = i - 1
        return False
```

#### 239.分糖果(575-Easy)

题意：给定一个偶数长度的数组，其中不同的数字代表着不同种类的糖果，每一个数字代表一个糖果。你需要把这些糖果平均分给一个弟弟和一个妹妹。返回妹妹可以获得的最大糖果的种类数。

分析：求妹妹分得的糖果种类数

```python
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        # 糖果数：len(candyType)
        # 糖果种类：len(set(candyType))
        # 妹妹分得一半糖果，且分到的糖果种类不超过总数
        return min(len(set(candyType)), len(candyType)//2)
```

#### 240.删除链表中的结点（237-Easy）

题意：请编写一个函数，用于删除单链表中某个特定节点。在设计函数时需要注意，你无法访问链表的头节点 head，只能直接访问要被删除的节点。题目数据保证需要删除的节点不是末尾节点。

分析：将待删除结点的下一个值赋给自己，然后将下一个结点删掉，如何让自己在世界上消失，但又不死？ —— 将自己完全变成另一个人，再杀了那个人就行了。

```python
class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        # 将待删除结点的下一个值赋给自己，然后将下一个结点删掉
        node.val = node.next.val
        node.next = node.next.next
```

#### 241.有效的完全平方数（367-Easy)

题意：给定一个 正整数 num ，编写一个函数，如果 num 是一个完全平方数，则返回 true ，否则返回 false 。进阶：不要 使用任何内置的库函数，如  sqrt 。

分析：找啊找啊找找找

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        # 二分查找
        left, right = 0, num
        while left <= right:
            mid = (left + right) // 2
            square = mid * mid
            if square < num:
                left = mid + 1
            elif square > num:
                right = mid - 1
            else:
                return True
        return False
```





------

## 字符串

#### 1.左旋转字符串（剑指OfferII58-Easy)

题目：https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/

分析：完全在本串上操作：局部反转+整体反转

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

#### 2.替换空格（剑指Offer05-Easy）

题意：https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/

分析：不用额外辅助空间，先扩充数组到填充后的大小，然后从后向前替换。

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



## 二叉树

#### 1.路径总和(112-Easy)

题目：https://leetcode-cn.com/problems/path-sum/

分析：

​	递归方法，遍历二叉树每次目标和减去遍历路径节点上的数值，如果最后target减去叶子节点的值为0，说明找到了目标和。

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        # 递归判断：遍历过程中每遇到一个结点，从目标值里扣除结点值，直到叶子节点判断目标值是否被扣完。
        if root == None:
            return False 
        # 遍历到叶子节点，判断当前所需目标和-叶子节点的值是否为零满足条件
        if root.left == None and root.right == None:
            return targetSum == root.val
        return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)
```

#### 2.路径总和II(113-Medium)

题目：https://leetcode-cn.com/problems/path-sum-ii/

分析：

​	采用深度优先搜索的方式，枚举每一条从根节点到叶子节点的路径。当我们遍历到叶子节点，且此时路径和恰为目标和时，我们就找到了一条满足条件的路径。

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        # 深度优先搜索:枚举根节点到叶子结点的路径，遍历到叶子结点恰好为目标和的时候，记录这条路径。
        res = []
        path = []
        def dfs(root: Optional[TreeNode], targetSum: int):
            if not root:
                return []
            path.append(root.val)
            targetSum -= root.val
            if not root.left and not root.right and targetSum == 0:
                res.append(path[:])
            dfs(root.left, targetSum)
            dfs(root.right, targetSum)
            # 清除当前选择，不影响其他路径搜索
            path.pop()
        dfs(root, targetSum)
        return res
```

#### 3.路径总和III(437-Meidum)

题目：https://leetcode-cn.com/problems/path-sum-iii/

分析：

​	前缀和加深度优先搜索，利用数组path存储到当前结点的前面所有前缀和，以路径上依次经过的结点作为根节点计算，然后在每次递归中统计path中等于targetSum的个数，先一直递归左子树然后一直递归右子树

```python
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:

        if not root:
            return 0
        self.res = 0

        def dfs(root, path):
            # 记录到root结点处的所有前缀路径和，以路径上每个都为根节点计算一个
            path = [val + root.val for val in path]
            path.append(root.val)
            # 计算当前路径和中含有多少个targetSum
            self.res += path.count(targetSum)
            # 递归左子树
            if root.left:
                dfs(root.left, path)
            # 递归右子树
            if root.right:
                dfs(root.right, path)
        # 递归起点
        dfs(root,[])
        return self.res
```



## 动态规划

#### 0.理论基础

**一、动态规划理论基础**

动规是由前一个状态推导出来的，而贪心是局部直接选最优。

1. 确定dp数组（dp table）以及下标的含义

2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组
找问题的最好方式就是把dp数组打印出来

动态规划的类型题：
（1）动规基础：斐波那契数列、爬楼梯
（2）背包问题
（3）打家劫舍
（4）股票问题
（5）子序列问题

**二、0-1背包理论基础(一）：二维dp数组**

​	背包问题：有N件物品和一个最多能被重量为W 的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。每件物品只能用一次，求解将哪些物品装入背包里物品价值总和最大。

**（1）确定dp数组及其下标含义**

​	`dp[i][j] `表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少。

**（2）确定递推公式**

​	不放物品i：由`dp[i-1][j]`推出。即背包容量为j，里面不放物品i的最大价值，此时`dp[i][j]`就是`dp[i-1][j]`。

​	放物品i：由`dp[i-1][j-weight[i]]`推出，`dp[i-1][j-weight[i]] `为背包容量为j - weight[i]的时候不放物品i的最大价值，那么`dp[i-1][j-weight[i]]` + value[i] （物品i的价值），就是背包放物品i得到的最大价值。

​	所以递归公式： `dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight[i]] + value[i])`;

**（3）dp数组如何初始化**

​	如果背包容量j为0的话，即`dp[i][0]`，无论是选取哪些物品，背包价值总和一定为0。`dp[0][j]`，即：i为0，存放编号0的物品的时候，各个容量的背包所能存放的最大价值，应该是value[0]如果j > weight[0]的话，j小于weight[0]那一定是0。

​	综上所述：一开始就统一把dp数组统一初始为0(i：物品个数，j：背包容量从0开始，背包数量+1），对于`dp[0][j]`在j > weight[0]的背包初始化为value[0]。

**（4）确定遍历顺序**

​	先遍历物品，后遍历背包重量，注意背包容量从小到大。

**三、0-1背包理论基础(二）：一维dp数组（滚动数组）**

​	滚动数组：需要满足的条件是上一层可以重复利用，直接拷贝到当前层。二维dp可以利用dp[i-1]拷贝到dp[i]上，简化为一维数组。

**（1）确定dp数组的定义**

​	在一维dp数组中，dp[j]表示：容量为j的背包，所背的物品价值可以最大为dp[j]。

**（2）一维dp数组的递推公式**

​	`dp[j] = max(dp[j], dp[j-weight[i]] + value[i]);`

​	dp[j]可以通过dp[j - weight[i]]推导出来，dp[j - weight[i]]表示容量为j - weight[i]的背包所背的最大价值，dp[j - weight[i]] + value[i] 表示 容量为j - 物品i重量的背包加上物品i的价值。此时dp[j]有两个选择，一个是取自己dp[j] 相当于二维dp数组中的`dp[i-1][j]`，即**不放物品i**，一个是取dp[j - weight[i]] + value[i]，即**放物品**i，指定是取最大的，毕竟是求最大价值。

**（3）一维dp数组如何初始化**

​	假设物品价值都是大于0的，所以dp数组初始化的时候，都初始为0。

**（4）一维dp遍历顺序**

​	先遍历物品，后遍历背包容量，背包容量从大到小。倒叙遍历是为了保证物品i只被放入一次

**四、完全背包理论基础**

​	完全背包和01背包问题唯一不同的地方就是，**每种物品有无限件**，01背包和完全背包唯一不同就是体现在遍历顺序上。01背包内嵌的循环是从大到小遍历，为了保证每个物品仅被添加一次，完全背包的物品是可以添加多次的，所以要从小到大去遍历。完全背包的遍历顺序二维dp物品在外层，背包在内层且背包容量，从小到大先后顺序可以颠倒，一维dp对于完全背包来说也可以颠倒。（只针对纯完全背包问题求得是能否凑成总和）

​	关键：内层背包遍历顺序从小到大，**如果求组合数就是外层for循环遍历物品，内层for遍历背包**。**如果求排列数就是外层for遍历背包，内层for循环遍历物品**。

​	[1,5]和[5,1]是同一种组合，但是是不同种排列

#### 1.斐波那契数（509-Easy）

题目：https://leetcode-cn.com/problems/fibonacci-number/

分析:

（1）状态定义：`dp[i]表示第i个数的斐波那契数值是dp[i]`;

（2）状态转移方程：`dp[i] = dp[i-1] + dp[i-2]`;

（3）dp数组初始化：题目已经给我们了

（4）遍历顺序：由递推式可以看出是从前到后遍历;

```python
class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        dp = [0] * (n+1)
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]
    
```

#### 2.爬楼梯（70-Easy）

题意：https://leetcode-cn.com/problems/climbing-stairs/

分析:

（1）状态定义：`dp[i]表示爬到第i层楼梯有的方法数`;

（2）状态转移方程：`dp[i] = dp[i-1] + dp[i-2]`;

（3）dp数组初始化：往结果上靠去解释dp[0] = 1，相当于直接站到楼顶。好像这样遍历就可以从2开始。但是其实可以不用考虑dp[0]，因为n是正整数

（4）遍历顺序：由递推式可以看出是从前到后遍历;

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        # 初始化成n+1是将n=2的情况包括进迭代中，以免特殊情况判断
        dp = [0] * (n+1)
        dp[0] = dp[1] = 1
        for i in range(2,n+1):
            # 第n个台阶只能从第n-1或者n-2个上来。
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]
```

#### 3.使用最小花费爬楼梯（746-Easy)

题意：https://leetcode-cn.com/problems/min-cost-climbing-stairs/

分析：

（1）状态定义：`dp[i]表示爬到第i层所需要的最低花费`;

（2）状态转移方程：`dp[i] = min(dp[i-1],dp[i-2]) + cost[i]`;

（3）dp数组初始化：根据递推公式可以看出初始化dp[0]和dp[1]即可，分别初始化为对应的cost[0]和cost[1]

（4）遍历顺序：由递推式可以看出是从前到后遍历;

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        
        N = len(cost)

        # d[i]表示爬到第i层需要的最低花费
        dp = [0] * N

        # 开始可以选择0或者1，在后面i=2的时候会判断选哪个更为最低
        dp[0] = cost[0]
        dp[1] = cost[1]

        for i in range(2, N):
            # dp[i]取决于之前爬一层上来还是爬两层上来最小值+当前花费
            dp[i] = min(dp[i-1],dp[i-2]) + cost[i]
        
        # N-1再踏一步就可以到楼顶，N-2再走两步就可以到楼顶，取最小花费
        return min(dp[N-1], dp[N-2])
```

#### 4.不同路径（62-Medium）

题意：https://leetcode-cn.com/problems/unique-paths/

分析：

（1）状态定义：`dp[i][j]表示从（0,0）出发，到（i,j)有dp[i][j]条不同的路径。`

（2）状态转移方程：`dp[i][j] = dp[i-1][j] + dp[i][j-1]`

（3）dp数组初始化：第一行和第一列都是1，因为从(0,0)到(i,0)和(0,j)的路径只有一条

（4）遍历顺序：从左到右一层层遍历就好

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 这题类似于64-最小路径和，采用动态规划
        # dp[i][j]表示从左上角到达右下角的路径总数
        dp = [[0] * n for _ in range(m)]
        # 第一列和第一行由于只有单方向所以路径数量一直为1
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        # 其他位置的路径数量等于来自两个方向的数量之和
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
```

#### 5.不同路径II（63-Medium）

题意：https://leetcode-cn.com/problems/unique-paths-ii/

分析：和不同路径的区分在于对于障碍的处理，有障碍的地方标记对应的dp为0即可

（1）状态定义：`dp[i][j]表示从（0,0）出发，到（i,j)有dp[i][j]条不同的路径。`

（2）状态转移方程：无障碍时`dp[i][j] = dp[i-1][j] + dp[i][j-1]`，有障碍则保持初始状态（初始状态为0）

（3）dp数组初始化：第一行和第一列无障碍处是1，有障碍处则是0而且障碍后面的都无法到达均为0。

（4）遍历顺序：从递推公式可以看出从左到右遍历

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # 与不同路径I区别的是：处理有障碍的地方dp应为0，第一行和第一列有障碍的位置后面全是0   

        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        # dp数组表示从(0,0)出发到(i,j)有dp[i][j]条不同的路径。
        dp = [[0] * n for _ in range(m)]

        # 初始位置为障碍则直接返回0条路径
        if obstacleGrid[0][0] == 1:return 0

        # 第一列和第一行，没有障碍更新dp，有障碍则跳出循环后面dp数组均为0
        for i in range(m):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = 1
            else:
                break
        for i in range(n):
            if obstacleGrid[0][i] == 0:
                dp[0][i] = 1
            else:
                break
        
        # 没有障碍更新dp来自两个方向，有障碍则继续循环该位置的dp为0
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:continue
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
```

#### 6.整数拆分（343-Medium）

题意：https://leetcode-cn.com/problems/integer-break/

分析：

（1）状态定义：`dp[i]：分拆数字i，可以得到的最大乘积为dp[i]。`

（2）状态转移方程：`dp[i] = max(dp[i], j * (i-j), j * dp[i-j])`，两种渠道得到dp[i]，j * (i - j) 是单纯的把整数拆分为两个数相乘，而j * dp[i - j]是拆分成两个以及两个以上的个数相乘。

（3）dp数组初始化：拆分0和1的最大乘积是无意义的，所以只初始化dp[2] = 1

（4）遍历顺序：dp[i] 是依靠 dp[i - j]的状态，所以遍历i一定是从前向后遍历，枚举j的时候，是从1开始的。i是从3开始，这样dp[i - j]就是dp[2]正好可以通过初始化的数值求出来。

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        # dp[i]分拆数字可以得到的最大乘积
        dp = [0] * (n + 1)
        # 0和1无解，所以只初始化dp[2]
        dp[2] = 1
        # i从3开始，j从1开始，dp[i-j]就是dp[2]可以通过初始化的值来求
        for i in range(3, n+1):
            for j in range(1, i):
                # 两种渠道得到dp[i]，拆分成j和i-j两个数相乘，j*dp[i-j]拆分成多个数相乘
                dp[i] = max(dp[i], j * (i-j), j * dp[i-j])
        return dp[n]
```

#### 7.不同的二叉搜索树（96-Medium）

题意：https://leetcode-cn.com/problems/unique-binary-search-trees/

分析：状态定义为`dp[i]表示由i个结点构成二叉搜索树的个数`，注意空树也是二叉树的一种，所以dp[0]=1，遍历所有1-n节点，分别计算当i作为根节点时对应的二叉树数量并求和，每种根节点对应的二叉树数量为其左子树数量乘以右子树数量。

（1）状态定义：`dp[i]：1到i为节点组成的二叉搜索树的个数为dp[i]`

（2）状态转移方程：`dp[i] += dp[j-1] * dp[i-j]`，j-1为j为头结点左子树节点数量，i-j为以j为头结点右子树节点数量

（3）dp数组初始化：dp[0] = 1，空节点也是一颗二叉树，也是一颗二叉搜索树

（4）遍历顺序：节点数为i的状态是依靠i之前节点数的状态，所以i是顺序遍历，j来遍历i里面每一个数作为头结点的状态。

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] *(n+1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n+1):
            # i个结点可以组成多少种二叉搜索树，分别以1-i为根节点计算左右子树可以组成的种类相乘
            for j in range(1, i+1):
                # j-1为j为头结点左子树节点数量，i-j为以j为头结点右子树节点数量
                dp[i] += dp[j-1] * dp[i-j]
        return dp[-1]
```

#### 8.分割等和子集（416-Medium）

题意：https://leetcode-cn.com/problems/partition-equal-subset-sum/

分析：0-1背包问题。如果dp[i] == i 说明，集合中的子集总和正好可以凑成总和i，理解这一点很重要

（1）状态定义：`dp[i]：背包总容量是i，最大可以凑成i的子集总和为dp[i]`

（2）状态转移方程：`dp[j] = max(dp[j], dp[j-nums[i]] + nums[i])`，分别为不放nums[i]和放nums[i]，物品i的重量是nums[i]，其价值也是nums[i]。

（3）dp数组初始化：初始化为0

（4）遍历顺序：如果使用一维dp数组，物品遍历的for循环放在外层，遍历背包的for循环放在内层，且内层for循环倒叙遍历。

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # 0-1背包问题

        # 数组和是奇数必然不能分割
        nums_sum = sum(nums)
        if nums_sum % 2 == 1:
            return False
        
        # 所求目标背包容量
        target = nums_sum // 2
        # dp[j]表示容量为j的背包所装数字之和最大为dp[j]
        dp = [0]* (target + 1)

        for i in range(len(nums)):
            for j in range(target, nums[i]-1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
            
        # 容量为target的背包所装数字之和必然要为target才可以分割成功
        return target == dp[target]
```

#### 9.最后一块石头的重量II（1049-Medium）

题意：https://leetcode-cn.com/problems/last-stone-weight-ii/

分析：0-1背包问题，尽量让石头分成重量相同的两堆，相撞之后剩下的石头最小

（1）状态定义：`dp[j]表示容量（这里说容量更形象，其实就是重量）为j的背包，最多可以背dp[j]这么重的石头`

（2）状态转移方程：`dp[j] = max(dp[j], dp[j-stones[i]] + stones[i])`，物品i的重量是stones[i]，其价值也是stones[i]。

（3）dp数组初始化：初始化为0，长度为重量总和的一半

（4）遍历顺序：如果使用一维dp数组，物品遍历的for循环放在外层，遍历背包的for循环放在内层，且内层for循环倒叙遍历。

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        # 0-1背包
        # 尽量让石头分成重量相同的两堆，相撞之后剩下的石头最小
        if len(stones) == 1:
            return stones[0]

        # dp[i]表示容量为i的情况下，最多能装多少重量的石头。
        target = sum(stones) // 2
        dp = [0] * (target + 1)

        # 先遍历石头重量，后逆序遍历背包容量
        for weight in stones:
            for j in range(target, weight - 1, -1):
                dp[j] = max(dp[j], dp[j - weight] + weight)
        # 将数据分成两组，每组最大重量为dp[sum/2]，最后剩最小的石头为sum - dp[sum/2] * 2
        return sum(stones) - dp[target] * 2
```

#### 10.目标和（494-Medium）

题意：https://leetcode-cn.com/problems/target-sum/

分析：利用0-1背包求组合问题，填满背包的方法数有多少，注意递推公式的变化：dp[j] += dp[j - nums[i]]，难点在于推导背包容量

（1）状态定义：`dp[j] 表示：填满j（包括j）这么大容积的包，有dp[j]种方法`

（2）状态转移方程：`dp[j] += dp[j - nums[i]]`，填满容量为j - nums[i]的背包，有dp[j - nums[i]]种方法，只要有nums[i]则凑成dp[j]也是dp[j-nums[i]]种方法。

（3）dp数组初始化：dp[0]=1，装满容量为0的背包，有1种方法，就是装0件物品，其他下标初始化为0。

（4）遍历顺序：nums放在外循环，target在内循环，且内循环倒序

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # x - (sum - x) = S, x = (S + sum) / 2
        # 0-1背包问题：装满x背包有几种方法，求组合类问题
        sumValue = sum(nums)
        if sumValue < abs(target) or (target + sumValue) % 2 == 1:
            return 0
        bagSize = (target + sumValue) // 2
        dp = [0] * (bagSize + 1)
        dp[0] = 1
        for i in range(len(nums)):
            for j in range(bagSize, nums[i]-1, -1):
                dp[j] += dp[j-nums[i]]
        return dp[-1]
```

#### 11.一和零（474-Medium）

题意：https://leetcode-cn.com/problems/ones-and-zeroes/

分析：两个维度的0-1背包问题，strs数组里的字符串是物品，里面的01个数相当于物品重量，字符串本身的个数相当于物品价值，m和n相当于一个两个维度的背包。

（1）状态定义：`dp[i][j]：最多有i个0和j个1的strs的最大子集的大小为dp[i][j]`

（2）状态转移方程：`dp[i][j] = max(dp[i][j], dp[i-zeroNum][j-oneNum] + 1);`，对比一下01背包的递推公式就会发现，字符串的zeroNum和oneNum相当于物品的重量（weight[i]），字符串本身的个数相当于物品的价值（value[i]）

（3）dp数组初始化：初始化为0

（4）遍历顺序：物品是strs里的字符串，所以外层正序遍历字符串，内层逆序遍历两个维度的背包容量，最大容量为题目中给的m和n。

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        # 两个维度的0-1背包问题
        dp = [[0] * (n+1) for _ in range(m+1)]
        for str in strs:
            # 01个数相当于统计每个物品的重量
            zero_num = str.count('0')
            one_num = str.count('1')
            for i in range(m, zero_num - 1, -1):
                for j in range(n, one_num - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zero_num][j - one_num] + 1)
        return dp[m][n]
```

#### 12.零钱兑换II（518-medium）

题意：https://leetcode-cn.com/problems/coin-change-2/

分析：钱币数量不限所以是完全背包问题，要求凑成总金额的个数。求组合数而不是排列，排列和元素顺序有关，组合和元素顺序无关

（1）状态定义：`dp[j]：凑成总金额j的货币组合数为dp[j]`

（2）状态转移方程：`dp[j] += dp[j - coins[i]]`，dp[j] （考虑coins[i]的组合总和） 就是所有的dp[j - coins[i]]（不考虑coins[i]）相加。

（3）dp数组初始化：dp[0] = 1，凑成总金额0的货币组合数为1，其他初始化为0

（4）遍历顺序：先遍历物品再遍历背包求组合数，先遍历背包再遍历物品求排列数，所以本题是求组合数所以先遍历钱币再遍历背包

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins:
            for j in range(coin, amount + 1):
                dp[j] += dp[j - coin]
        return dp[-1]
```

#### 13.组合总和IV（377-Medium）

题意：https://leetcode-cn.com/problems/combination-sum-iv/

分析：注意示例里面说顺序不同的序列被视作不同的组合，所以本质上是在求排列总和的个数

（1）状态定义：`dp[i]: 凑成目标正整数为i的排列个数为dp[i]`

（2）状态转移方程：`dp[i] += dp[i - nums[j]]`

（3）dp数组初始化：dp[0] = 1，其他初始化为0

（4）遍历顺序：target（背包）放在外循环，将nums（物品）放在内循环，内循环从前到后遍历。注意要判断背包要大于物品重量

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i-num]
        return dp[-1]
```

#### 14.零钱兑换（322-Medium）

题意：https://leetcode-cn.com/problems/coin-change/

分析：

（1）状态定义：`dp[i]: 凑足总额为j所需钱币的最少个数为dp[i]`

（2）状态转移方程：`dp[i] = min(dp[i - coins[i]] + 1, dp[i])`

（3）dp数组初始化：dp[0] = 0，其他初始化为float('inf')

（4）遍历顺序：coins（物品）放在外循环，target（背包）在内循环。且内循环正序。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # 完全背包
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i-coin] + 1)
        if dp[amount] == float('inf'):
            return -1
        else:
            return dp[amount]
```

#### 15.完全平方数(279-Medium)

题意：https://leetcode-cn.com/problems/perfect-squares/

分析：完全平方数就是物品（可以无限件使用），凑个正整数n就是背包，问凑满这个背包最少有多少物品？

（1）状态定义：`dp[i]: 和为i的完全平方数的最少数量为dp[i]`

（2）状态转移方程：`dp[j] = min(dp[j - i * i] + 1, dp[j])`

（3）dp数组初始化：dp[0] = 0，非0下标的dp[i]一定要初始为最大值i（i个1）

（4）遍历顺序：完全背包，外层背包内层物品均可以

```python
class Solution:
    def numSquares(self, n: int) -> int:
        # 完全背包问题：完全平方数是物品，和为背包
        # 组成和的完全平方数的最多个数，就是只用1构成
        dp = [i for i in range(n + 1)]

        # 先遍历物品还是背包都可以
        # 这里先遍历背包再遍历物品
        for i in range(1, n+1):
            for j in range(1, n):
                num = j * j
                # 背包容量要大于物品，如果小于直接break
                if num > i:break
                if i - num >= 0:
                    dp[i] = min(dp[i], dp[i-num] + 1)
        return dp[n]
    	# 先遍历物品后遍历背包
        for i in range(1, n):
            if i * i > n:
                break
            num = i * i
            for j in range(num, n+1):
                dp[j] = min(dp[j], dp[j-num] + 1)
        return dp[n]
```

#### 16.单词拆分(139-Medium)

题意：https://leetcode-cn.com/problems/word-break/

分析：单词就是物品，字符串s就是背包，单词能否组成字符串s，就是问物品能不能把背包装满

（1）状态定义：`dp[i] : 字符串长度为i的话，dp[i]为true，表示可以拆分为一个或多个在字典中出现的单词`

（2）状态转移方程：`if([j, i] 这个区间的子串出现在字典里 && dp[j]是true) 那么 dp[i] = true`

（3）dp数组初始化：dp[0] = true，是递归的基础

（4）遍历顺序：要求子串，最好是遍历背包放在外循环，将遍历物品放在内循环，内循环从前到后。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # 完全背包，dp[i]为true，表示长度为i的字符串可以拆分为一个或多个在字典中出现的单词
        N = len(s)
        dp = [False] * (N + 1)
        dp[0] = True
        # 先遍历背包后遍历物品
        for i in range(1, N+1):
            for j in range(i):
                # dp[j] 是true，且 [j, i] 这个区间的子串出现在字典里，那么dp[i]一定是true
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
        return dp[-1]
```





## 双指针法

#### 8.环形链表II（142-Medium）

题意：给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 `null`。

分析：本题与环形链表I不同的地方在于，环形链表I仅需判断是否存在环（利用快慢指针入环相遇来判断），而环形链表II则需要判断链表开始入环的第一个结点。在寻找环入口的过程中通过数学推理可以知道：**从头结点出发一个指针，从相遇节点也出发一个指针，这两个指针每次只走一个节点，那么当这两个指针相遇的时候就是环形入口的节点**。清楚这两点即可解决这道题了，首先找到在环内快慢指针相遇的结点，然后利用数学推理找到环形入口的结点。

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        # 判断是否存在环，找到环相遇的结点
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                # 从头结点和相遇结点出发，各寻找环的入口结点
                p = head
                q = slow
                while p != q:
                    p = p.next
                    q = q.next
                return p
        return None
```

## 单调栈

#### 1.每日温度(739-Medium)

题意：请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

分析：这是第一次独立完成的单调栈题目，遇到需要寻找左边或者右边比其大的数可以考虑单调栈。建立一个从栈底到栈顶的单调递减栈，遍历数组元素，栈存在且当前元素大于栈内元素时，计算当前元素位置与栈内已存储的元素位置的距离（此时栈内元素都是比当前元素小的值，所以计算到当前位置的等待天数），存入结果数组，再将当前遍历的元素下标存入栈中用于下一次计算。

```python
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        N = len(T)
        stack = []
        res = [0] * N
        # 建立单调递减栈存储下标
        for i in range(N):
            # 当栈存在且当前元素大于栈顶元素时，即要计算与栈内与其小的元素依次计算距离差值放入结果数组中
            while stack and T[stack[-1]] < T[i]:
                res[stack[-1]] = i - stack[-1]
                stack.pop()
            # 栈不存在或者当前元素小于栈顶元素将其入栈，寻找后面比他们大的元素
            stack.append(i)
        return res
```

#### 2.下一个更大元素I（496-Easy)

题意：给你两个 没有重复元素 的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。请你找出 nums1 中每个元素在 nums2 中的下一个比其大的值。nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出 -1 。

分析：单调栈，从栈顶到栈底保持递增顺序，主要处理nums2的元素，遍历nums2。当前元素大于栈顶时，判断当前元素是否在nums1内，如果在的话则索引nums1的位置，然后记录当前元素。当前元素小于等于栈顶时，直接入栈。栈内存储元素在nums2中的下标。

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 单调栈，构造从栈底到栈顶的单调递增栈
        stack = []
        res = [-1] * len(nums1)
        for i in range(len(nums2)):
            # 当前元素大于栈顶时，判断当前元素是否在nums1内，如果在的话则索引nums1的位置，然后记录当前元素
            while stack and nums2[i] > nums2[stack[-1]]:
                if nums2[stack[-1]] in nums1:
                    index = nums1.index(nums2[stack[-1]])
                    res[index] = nums2[i]
                stack.pop()
            # 当前元素小于等于栈顶时，直接入栈
            stack.append(i)
        return res
    # 单调栈里存元素的情况
     def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dist, stack = {}, []
        for n in nums2:
            while stack and stack[-1] < n:
                dist[stack.pop()] = n
            stack.append(n)

        res = []
        for n in nums1:
            res.append(dist.get(n, -1))
        return res
```

#### 3.下一个更大元素II(503-Medium)

题意：给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 x 的下一个更大的元素是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1。

分析：如何求下一个更大元素（单调栈）+如何实现循环数组（取模）

（1）求下一个更大元素：暴力求解每个元素会超时，可以遍历一次数组，如果元素是单调递减的（则他们的「下一个更大元素」相同），我们就把这些元素保存，直到找到一个较大的元素；把该较大元素逐一跟保存了的元素比较，如果该元素更大，那么它就是前面元素的「下一个更大元素」，使用单调栈，**单调栈是说栈里面的元素从栈底到栈顶是单调递增或者单调递减的（类似于汉诺塔）**，建立单调递减栈并对原数组遍历一次，栈内存储数组下标因为结果数组需要用到，元素下标依次入栈，栈非空时比较当前元素和栈顶元素的大小，当前元素大于栈顶元素，则存入结果数组并弹出栈顶元素直到当前元素比栈顶元素小，当前元素小于栈顶元素将当前元素入栈，栈内元素对应的下一个更大元素相同。

（2）实现循环数组：循环数组即为数组最后一个元素下一个元素是数组的第一个元素，取模运算将下标i映射到数组长度的0-N范围内，本题i取到2N-1。

```python
class Solution(object):
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 第一种方法：建立单调递减栈
        N = len(nums)
        res = [-1] * N
        # 利用栈存储数组下标，因为需要根据下标修改结果数组
        stack = []
        for i in range(N * 2):
            # 栈非空的时候判断栈顶元素下标对应数值和当前数值比较
            while stack and nums[stack[-1]] < nums[i % N]:
                # 满足条件弹出栈顶元素，根据下标记录到结果数组中
                res[stack.pop()] = nums[i % N]
            # 当前数值小于栈顶元素下标对应数值，入栈建立单调递减关系，继续寻找下一个最大的值
            # 当前栈内元素均小于栈顶元素，所以下一个最大的值均是当前栈内
            stack.append(i % N)
        return res
```

#### 4.接雨水(42-Hard)

题意：给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png" alt="img"  />

分析：单调栈，从栈头（元素从栈头弹出）到栈底的顺序应该是从小到大的顺序，一旦发现添加的柱子高度大于栈头元素了，此时就出现凹槽了，栈头元素就是凹槽底部的柱子，栈头第二个元素就是凹槽左边的柱子，而添加的元素就是凹槽右边的柱子。**关键：找每个柱子左右两边第一个大于该柱子高度的柱子**，决定单调栈的顺序从栈顶到栈底的顺序是单调递增。

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # 单调栈存储下标
        res = 0
        stack = []
        N = len(height)
        # 维护单调栈：从栈顶到栈底是从小到大顺序，当前h大于栈顶元素时出现凹槽，计算体积
        for i, h in enumerate(height):
            while stack and h > height[stack[-1]]:
                mid = stack.pop()
                # 注意如果出现弹出后栈内为空，则左边无柱子构不成凹槽直接将当前i入栈，将左侧更新为当前i的高度
                if not stack:
                    break
                left = stack[-1]

                currWidth = i - left - 1
                # 当前栈顶元素为中间凹槽底部的柱子下标，pop之后下一个栈顶就是其左边柱子下标，当前i为右边柱子下标
                currHeight = min(height[left],height[i]) - height[mid]
                # 计算当前位置体积累加到res中
                res += currWidth * currHeight
            stack.append(i)
        return res

```

#### 5.柱状图中最大的矩形（84-Hard）

题意：给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。求在该柱状图中，能够勾勒出来的矩形的最大面积。<img src="https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg" alt="img"  />

分析：因为本题是要找每个柱子左右两边第一个小于该柱子的柱子，所以从栈头（元素从栈头弹出）到栈底的顺序应该是从大到小的顺序！**栈顶和栈顶的下一个元素以及要入栈的三个元素组成了我们要求最大面积的高度和宽度**。求以第i根柱子为最矮柱子所能延伸的最大面积。

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 单调栈：从栈顶到栈底单调递减
        # 找每个柱子左右两边第一个小于该柱子的柱子
        heights = [0] + heights + [0]
        stack = []
        res = 0
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                mid = stack.pop()
                # 计算以第i根柱子为最矮柱子所能延伸的最大面积
                res = max(res, (i - stack[-1] - 1) * heights[mid])
            stack.append(i)
        return res
```



