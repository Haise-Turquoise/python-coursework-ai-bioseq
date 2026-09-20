# CS482 Assignment 1 - Fit Alignment
# Name: Yuanhao Liu (Jim Liu)
# Student ID: 20775207
# Date: 2026-02-01

import sys

# ===== Score Scheme =====
MATCH = 1
MISMATCH = -1
INDEL = -1

# ===== Helper Funcs =====

def readFasta(filename):
    """
    读取简化版FASTA文件
    read simplified FASTA file, return two sequences
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 去掉换行符 / strip newlines
    lines = [line.strip() for line in lines]
    
    # 第2行和第4行是序列 / 2nd and 4th lines are sequences
    seqS = lines[1]
    seqT = lines[3]
    
    return seqS, seqT

def getScore(a, b):
    """
    返回两个字符的对齐分数
    return alignment score for two characters
    """
    if a == b:
        return MATCH
    else:
        return MISMATCH

# ===== DP Functions =====

def buildDPTable(S, T):
    """
    构建DP表,返回填好的表
    build and fill DP table, return the table
    """
    m = len(S)  # S的长度
    n = len(T)  # T的长度
    
    # 创建(m+1) x (n+1)的表 / create table
    D = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化第一列 / init first column
    for i in range(1, m + 1):
        D[i][0] = i * INDEL
    
    # 初始化第一行 / init first row
    for j in range(1, n + 1):
        D[0][j] = 0
    
    # 填表 / fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            fromDiag = D[i-1][j-1] + getScore(S[i-1], T[j-1])  # 注意: Python下标从0开始
            fromUp = D[i-1][j] + INDEL
            fromLeft = D[i][j-1] + INDEL
            D[i][j] = max(fromDiag, fromUp, fromLeft)
    
    return D

def findAnswer(D, m, n):
    """
    找到答案的分数和位置
    find answer score and position
    """
    pendMax = -1000000
    pendidx = -1
    for i in range(n+1):
        if pendMax < D[m][i]: #if same choose the first occur max score
            pendidx = i
            pendMax = D[m][i]
    score = pendMax
    endJ = pendidx  # backtrack起点的j坐标
    
    return score, endJ

def backtrack(D, S, T, endJ):
    """
    回溯得到对齐结果
    backtrack to get alignment strings
    """
    m = len(S)
    n = len(T)
    
    alignS = ""
    alignT = ""
    
    i = m
    j = endJ  
    
    while i > 0 :
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1] + getScore(S[i-1], T[j-1]):
            # 来自对角线 / from diagonal
            alignS = S[i-1] + alignS
            alignT = T[j-1] + alignT
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i-1][j] + INDEL:
            # 来自上方 / from up
            alignS = S[i-1] + alignS
            alignT = "-" + alignT
            i -= 1
        else:
            # 来自左方 / from left
            alignS = "-" + alignS
            alignT = T[j-1] + alignT
            j -= 1
    
    return alignS, alignT

def writeOutput(filename, score, alignS, alignT):
    """
    写入输出文件
    write output file
    """
    with open(filename, 'w') as f:
        f.write(str(score) + "\n")
        f.write(alignS + "\n")
        f.write(alignT + "\n")

# ===== Main =====

def main():
    # 读取命令行参数 / read command line args
    if len(sys.argv) != 3:
        print("Usage: python assn1-part1.py in.fasta out.txt")
        return
    
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    
    # 读取序列 / read sequences
    S, T = readFasta(inFile)
    
    # 构建DP表 / build DP table
    D = buildDPTable(S, T)
    
    # 找答案 / find answer
    m, n = len(S), len(T)
    score, endJ = findAnswer(D, m, n)
    
    # 回溯 / backtrack
    alignS, alignT = backtrack(D, S, T, endJ)
    
    # 输出 / output
    writeOutput(outFile, score, alignS, alignT)
    

if __name__ == "__main__":
    main()