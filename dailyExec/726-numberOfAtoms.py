'''
Description: 原子的数量【困难】
Author: Luminary
Date: 2021-07-05 12:01:11
LastEditTime: 2021-07-05 12:01:52
'''
from collections import defaultdict
class Solution:
    def countOfAtoms(self, formula) :
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

