


import sys

# ===== Sorting Helpers =====

def singleFreqSort(idxArr, keyList, keyRange):
    # counting sort on idxArr, key = keyList[i], range 0..keyRange
    # keyRange: ~27 for first char pass, ~n after that

    freqArr = [0] * (keyRange + 1)
    for i in idxArr:
        freqArr[keyList[i]] += 1          # count frequency of each key value

    prefixArr = [0] * (keyRange + 1)
    for k in range(1, keyRange + 1):
        prefixArr[k] = prefixArr[k - 1] + freqArr[k - 1]   # each slot = start pos for this key

    singleSortedArr = [0] * len(idxArr)
    for i in idxArr:            # reverse pass keeps stable order
        k = keyList[i]
        singleSortedArr[prefixArr[k]] = i
        prefixArr[k] += 1

    return singleSortedArr


def dualFreqSort(idxArr, rankArr, L, curLen):
    # two-pass radix sort on (rankArr[i], rankArr[i+L])
    # pass 1: sort by second component rank[i+L], out-of-bound -> 0 (sentinel rank)
    # pass 2: first component rank[i]

    secondKeyList = []
    for i in range(curLen):
        if i + L < curLen:
            secondKeyList.append(rankArr[i + L])
        else:
            secondKeyList.append(0)    # out-of-bound -> sentinel rank

    idxArr = singleFreqSort(idxArr, secondKeyList, curLen)   # pass 1: second component
    idxArr = singleFreqSort(idxArr, rankArr,       curLen)   # pass 2: first  component

    return idxArr




# ===== Init Block (L=1 sort + first renaming) =====

def build_suffix_array(s):
    s = s + '$'                          # append sentinel, lexicographically smallest
    curLen  = len(s)
    charRange = 27                       # '$'=0, 'a'=1 ... 'z'=26

    # map each position to char key, '$' -> 0, letters -> 1-26
    charKeyList = []
    for ch in s:
        if ch == '$':
            charKeyList.append(0)
        else:
            charKeyList.append(ord(ch) - ord('a') + 1)

    # initial sort by single char key (L=1 pass)
    idxArr  = list(range(curLen))
    idxArr  = singleFreqSort(idxArr, charKeyList, charRange)

    # first renaming: assign rank 0..k based on sorted char order
    # same char as previous -> same rank, different -> rank+1
    rankArr = [0] * curLen
    rankArr[idxArr[0]] = 0
    rankMax = 0
    for pos in range(1, curLen):
        prevIdx = idxArr[pos - 1]
        currIdx = idxArr[pos]
        if s[currIdx] != s[prevIdx]:     # compare original chars, not charKey
            rankMax += 1
        rankArr[currIdx] = rankMax


# ===== Core Algorithm Block (prefix doubling main loop) =====

    L = 1
    while rankMax < curLen - 1 and L < curLen:

        # radix sort step: pairs (rankArr[i], rankArr[i+L])
        idxArr = dualFreqSort(idxArr, rankArr, L, curLen)

        # renaming pass, write to newRankArr (not in-place)
        newRankArr = [0] * curLen
        newRankArr[idxArr[0]] = 0
        rankMax = 0

        for pos in range(1, curLen):
            prevIdx = idxArr[pos - 1]
            currIdx = idxArr[pos]

            # get pair values, i+L out of range -> 0
            
            prevFirstRank  = rankArr[prevIdx]
            prevSecondRank = rankArr[prevIdx + L] if prevIdx + L < curLen else 0
            currFirstRank  = rankArr[currIdx]
            currSecondRank = rankArr[currIdx + L] if currIdx + L < curLen else 0

            if currFirstRank != prevFirstRank or currSecondRank != prevSecondRank:
                rankMax += 1                  # pair differs from previous -> new rank
            newRankArr[currIdx] = rankMax

        rankArr = newRankArr
        
        L = L * 2                             # next round covers 2x length

    return idxArr


# ===== IO Block =====

if __name__ == '__main__':
    s = sys.stdin.read().strip()
    if not s:                            # guard against accidental empty input
        sys.exit(0)
    resultArr = build_suffix_array(s)
    print(' '.join(str(i) for i in resultArr))