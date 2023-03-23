'''
Description: 十进制转M进制
Author: Luminary
Date: 2021-09-02 19:53:47
LastEditTime: 2021-09-02 21:01:02
'''

'''
# 将10进制数字N，交换为M进制数，计算并返回变换后的各位数字之和.(其中N<10000 2<=M<= 10)
    输入N, M:9, 5
    输出:5
    解释: 9变换为5进制后为1 4，1+4=5
    输入N, M:8, 8
    输出: 1
    解释: 8变换后8进制后为10, 1+0=1
    样例输入:95
    样例输出:5
'''
# 十进制转换 M进制输出所得数字的和
def func_toBase_sum (N, M):
    # n为待转换的十进制数，x为机制，取值为 2-10
    res = []
    while True:
        quotient = N // M  # 商
        remainder = N % M  # 余数
        res = res + [remainder]
        if quotient == 0:
            break
        N = quotient
    # 输出转换后的进制和
    print(sum(res))

# 十进制转换X进制（2-16），输出转换之后的数字
def func_toBase_output (N, M):
    # n为待转换的十进制数，x为机制，取值为 2-16
    res = []
    base = [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f']
    while True:
        quotient = N // M  # 商
        remainder = N % M  # 余数
        res = res + [remainder]
        if quotient == 0:
            break
        N = quotient
    res.reverse()
    # 输出转换后的进制
    for i in res:
        print(base[i], end='')

func_toBase_sum(9, 5)
func_toBase_output(9, 5)