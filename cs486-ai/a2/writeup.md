# CS 486/686: Assignment 2

---

## **1. Bayesian Networks and Variable Elimination (40 points)**

### **Q1a [10 pts]**

[TODO: Bayes net graph + CPT tables]

### **Q1b [25 pts]**

[TODO: VE factor tables for P(Fraud) and P(Fraud|FP, not IP, CRP)]

### **Q1c [5 pts]**

[TODO: stolen card strategy]

---

## **2. Markov Decision Processes (60 points)**

### **Q2a [20 pts]**

R(s) = -1.0 for all non-terminal states. gamma = 1. Value iteration until convergence (3 decimal places).

Utility values per iteration:

```
Iteration 0:
   0.000     0.000     0.000     0.000
   0.000     X         0.000    -1.000
   0.000     0.000     0.000     1.000

Iteration 1:
  -1.000    -1.000    -1.000    -1.000
  -1.000     X        -1.000    -1.000
  -1.000    -1.000    -0.200     1.000

Iteration 2:
  -2.000    -2.000    -2.000    -2.000
  -2.000     X        -1.360    -1.000
  -2.000    -1.360    -0.320     1.000

Iteration 3:
  -3.000    -3.000    -2.488    -2.200
  -3.000     X        -1.492    -1.000
  -2.488    -1.528    -0.368     1.000

Iteration 4:
  -4.000    -3.590    -2.714    -2.269
  -3.590     X        -1.544    -1.000
  -2.771    -1.600    -0.386     1.000

Iteration 5:
  -4.631    -3.889    -2.821    -2.298
  -3.935     X        -1.563    -1.000
  -2.916    -1.629    -0.393     1.000

Iteration 6:
  -4.968    -4.034    -2.869    -2.312
  -4.120     X        -1.571    -1.000
  -2.988    -1.640    -0.396     1.000

Iteration 7:
  -5.136    -4.102    -2.891    -2.318
  -4.215     X        -1.574    -1.000
  -3.023    -1.645    -0.397     1.000

Iteration 8:
  -5.217    -4.133    -2.901    -2.321
  -4.261     X        -1.575    -1.000
  -3.039    -1.646    -0.397     1.000

Iteration 9:
  -5.255    -4.147    -2.905    -2.322
  -4.284     X        -1.575    -1.000
  -3.047    -1.647    -0.397     1.000

Iteration 10:
  -5.272    -4.154    -2.907    -2.323
  -4.294     X        -1.575    -1.000
  -3.051    -1.647    -0.397     1.000

Iteration 11:
  -5.279    -4.156    -2.908    -2.323
  -4.299     X        -1.575    -1.000
  -3.052    -1.647    -0.397     1.000

Iteration 12:
  -5.283    -4.158    -2.908    -2.323
  -4.302     X        -1.575    -1.000
  -3.053    -1.647    -0.397     1.000

Iteration 13:
  -5.284    -4.158    -2.908    -2.323
  -4.303     X        -1.575    -1.000
  -3.053    -1.647    -0.397     1.000

Iteration 14:
  -5.285    -4.158    -2.908    -2.323
  -4.303     X        -1.575    -1.000
  -3.053    -1.647    -0.397     1.000

Iteration 15 (converged - same as iteration 14):
  -5.285    -4.158    -2.908    -2.323
  -4.303     X        -1.575    -1.000
  -3.053    -1.647    -0.397     1.000
```

Optimal policy (R(s) = -1.0):

|       | col1 | col2 | col3 | col4 |
|-------|------|------|------|------|
| row1  | →    | →    | ↓    | ↓    |
| row2  | ↓    | X    | ↓    | -1   |
| row3  | →    | →    | →    | +1   |

### **Q2b [20 pts]**

Three policy ranges found. Rows 2-3 stay the same across all R values — only row 1 changes.

**Range 1: R(s) = -1.6**

|       | col1 | col2 | col3 | col4 |
|-------|------|------|------|------|
| row1  | →    | →    | →    | ↓    |
| row2  | ↓    | X    | ↓    | -1   |
| row3  | →    | →    | →    | +1   |

s(1,3) = → — agent at (1,3) heads right toward s(1,4) then down to the -1 terminal. Living cost is so high (-1.6/step) that reaching any terminal fast beats a longer path to +1.

**Transition: R = -1.6 → R = -1.5**
s(1,3) changes from → to ↓. Agent now goes down toward +1 instead of right toward -1.

**Range 2: R(s) = -1.5 to -0.8**

|       | col1 | col2 | col3 | col4 |
|-------|------|------|------|------|
| row1  | →    | →    | ↓    | ↓    |
| row2  | ↓    | X    | ↓    | -1   |
| row3  | →    | →    | →    | +1   |

s(1,1) = → — still heading right, taking a slightly longer path toward +1 via row 3.

**Transition: R = -0.8 → R = -0.7**
s(1,1) changes from → to ↓. Agent now goes directly down — shorter path to row 3 and the +1 terminal.

**Range 3: R(s) = -0.7 to -0.5**

|       | col1 | col2 | col3 | col4 |
|-------|------|------|------|------|
| row1  | ↓    | →    | ↓    | ↓    |
| row2  | ↓    | X    | ↓    | -1   |
| row3  | →    | →    | →    | +1   |

All non-terminal states now route toward the +1 terminal. Lower living penalty gives the agent more room to avoid -1.

### **Q2c [20 pts]**

Two policy ranges found. Again only row 1 changes — rows 2-3 identical across all R values.

**Range 1: R(s) = -0.08 to -0.05**

|       | col1 | col2 | col3 | col4 |
|-------|------|------|------|------|
| row1  | ↓    | ←    | ↓    | ←    |
| row2  | ↓    | X    | ↓    | -1   |
| row3  | →    | →    | →    | +1   |

s(1,3) = ↓ — goes down toward row 3 and the +1 terminal. s(1,4) = ← — actively moves away from the -1 terminal. Living penalty is small enough that the agent takes longer paths to avoid -1.

**Transition: R = -0.05 → R = -0.04**
s(1,3) changes from ↓ to ←. Agent shifts from going down to going left — even more avoidance of the -1 side of the grid.

**Range 2: R(s) = -0.04 to -0.03**

|       | col1 | col2 | col3 | col4 |
|-------|------|------|------|------|
| row1  | ↓    | ←    | ←    | ←    |
| row2  | ↓    | X    | ↓    | -1   |
| row3  | →    | →    | →    | +1   |

s(1,3) now also ← — nearly zero living cost, so the agent strongly prefers the longest safe route to +1 over any path near -1.

---

## **3. Bonus: Viterbi Decoding (20 extra points)**

Solved by program — see code.py.

HMM parameters: P(s0)=0.4, P(st|st-1)=0.7, P(st|not st-1)=0.2, P(ot|st)=0.9, P(ot|not st)=0.2.
Observations: O0=T, O1=T, O2=T, O3=F.

Step-by-step delta values and back-pointers:

| Step | S=True delta | S=True backptr | S=False delta | S=False backptr |
|------|-------------|----------------|---------------|-----------------|
| 0    | 0.36        | -              | 0.12          | -               |
| 1    | 0.2268      | True           | 0.0216        | True            |
| 2    | 0.142884    | True           | 0.013608      | True            |
| 3    | 0.01000188  | True           | 0.03429216    | True            |

Termination: max(0.01000188, 0.03429216) = 0.03429216 at S3=False.

Traceback: S3=F, backptr=T -> S2=T, backptr=T -> S1=T, backptr=T -> S0=T.

**Optimal hidden state sequence: (T, T, T, F)**
