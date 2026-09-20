# CS 486/686: Assignment 3

course name: CS486
name: Yuanhao Liu (Jim Liu)
uw email: y2544liu@uwaterloo.ca
quest id: y2544liu

---

## Question 1 [35pts]: Text Categorization with Decision Trees

### Decision Trees (10 internal nodes)

#### Method 1 (evenly weighted IG)

```
christian (IG=0.500206)
  yes:
    [atheism] (35 docs)
  no:
    atheism (IG=0.500193)
      yes:
        [atheism] (24 docs)
      no:
        christians (IG=0.499876)
          yes:
            [atheism] (19 docs)
          no:
            beliefs (IG=0.499554)
              yes:
                [atheism] (19 docs)
              no:
                atheists (IG=0.498766)
                  yes:
                    [atheism] (12 docs)
                  no:
                    brain (IG=0.498247)
                      yes:
                        [atheism] (10 docs)
                      no:
                        aa (IG=0.497675)
                          yes:
                            [atheism] (7 docs)
                          no:
                            murder (IG=0.497345)
                              yes:
                                [atheism] (7 docs)
                              no:
                                proof (IG=0.496930)
                                  yes:
                                    [atheism] (6 docs)
                                  no:
                                    logic (IG=0.496601)
                                      yes:
                                        [atheism] (6 docs)
                                      no:
                                        [books] (1355 docs)
```

Solid line = word present (yes), dashed line = word absent (no).

Method 1 produces a degenerate chain — each split peels off a small group of atheism docs from one side, leaving nearly all docs in the other child. IG values are all close to 0.5 because the (1/2, 1/2) weighting inflates gains from lopsided splits.

#### Method 2 (fraction-weighted IG)

```
book (IG=0.077019)
  yes:
    bible (IG=0.115358)
      yes:
        [atheism] (4 docs)
      no:
        call (IG=0.069459)
          yes:
            [atheism] (2 docs)
          no:
            sent (IG=0.084224)
              yes:
                [atheism] (2 docs)
              no:
                controlling (IG=0.058758)
                  yes:
                    [atheism] (1 docs)
                  no:
                    [books] (146 docs)
  no:
    books (IG=0.059926)
      yes:
        sure (IG=0.097695)
          yes:
            soon (IG=0.918296)
              yes:
                [books] (1 docs)
              no:
                [atheism] (2 docs)
          no:
            spirit (IG=0.096945)
              yes:
                [atheism] (1 docs)
              no:
                [books] (79 docs)
      no:
        religion (IG=0.035673)
          yes:
            [atheism] (65 docs)
          no:
            [atheism] (1197 docs)
```

Method 2 produces a proper tree structure — root splits on "book" into two meaningful subtrees. IG values are smaller but more realistic. The fraction weighting penalizes lopsided splits correctly.


### Accuracy Curves

#### Method 1

![Method 1 Accuracy](q1_method1_accuracy.png)

Final accuracy at 100 nodes — Train: 75.67%, Test: 58.80%.

Train accuracy keeps climbing, test accuracy plateaus around 57-59% after ~15 nodes. The growing gap = overfitting, consistent with Method 1 picking features that separate tiny groups.

#### Method 2

![Method 2 Accuracy](q1_method2_accuracy.png)

Final accuracy at 100 nodes — Train: 83.40%, Test: 69.93%.

Both train and test accuracy are higher than Method 1. Test accuracy reaches ~70% and stays more stable — fraction-weighted IG picks more informative splits.


---

## Question 2 [65pts]: Neural Networks

### Implementation

Code in neural_net.py and operations.py. All public tests pass (10/10).

### K-Fold Cross Validation (k=5)

**Network architecture:** [16, 1] — one hidden layer of 16 neurons with Sigmoid activation, output layer of 1 neuron with Identity activation.

**Hyperparameters:** learning rate η = 0.001, 500 epochs.

**Results:**

| Fold | Validation MAE |
|------|----------------|
| 1    | 0.6546         |
| 2    | 0.6574         |
| 3    | 0.6925         |
| 4    | 0.6980         |
| 5    | 0.5985         |

**Average MAE: 0.6602 ± 0.0356**

#### Average Training Loss

![K-Fold Training Loss](q2_kfold_loss.png)

Loss drops from ~31 to ~0.6 in the first 150 epochs, then stabilizes. Standard convergence behavior for batch gradient descent with a small learning rate.


---

## Bonus Question [40pts]: Training with PyTorch

Code in train_model_pytorch.py.

**Network architecture:** Linear(11, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1).

**Setup:** 90/10 train/validation split (seed=42), MSELoss, Adam optimizer, lr=0.001, 500 epochs.

**Final validation MAE: 0.5147**

### Validation MAE vs Epoch

![PyTorch Validation MAE](q3_val_mae.png)

MAE drops from ~4.0 to ~0.6 in the first 80 epochs, then gradually decreases to 0.51. The deeper network + Adam optimizer achieves lower MAE than the from-scratch implementation (0.66 with k-fold).
