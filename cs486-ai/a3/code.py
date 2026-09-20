import numpy as np
import heapq
import matplotlib.pyplot as plt


# === data loading ===
def load_bow(data_path, label_path, word_path):
    labels = np.loadtxt(label_path, dtype=int)
    with open(word_path, encoding='utf-8') as f:
        words = [line.strip() for line in f]
    n_docs, n_words = len(labels), len(words)
    pairs = np.loadtxt(data_path, dtype=int)
    feat = np.zeros((n_docs, n_words), dtype=np.int8)
    feat[pairs[:, 0] - 1, pairs[:, 1] - 1] = 1
    return feat, labels, words


# === entropy ===
def ent(c1, c2):
    n = c1 + c2
    if n == 0:
        return 1.0
    h = 0.0
    for c in [c1, c2]:
        if c > 0:
            p = c / n
            h -= p * np.log2(p)
    return h


def ent_vec(c1, c2):
    n = c1 + c2
    out = np.ones(len(c1))
    nz = n > 0
    h = np.zeros(len(c1))
    for c in [c1, c2]:
        p = np.zeros(len(c1))
        p[nz] = c[nz] / n[nz]
        m = p > 0
        h[m] -= p[m] * np.log2(p[m])
    out[nz] = h[nz]
    return out


# === best feature for splitting a leaf ===
def find_best(feat, labels, docs, method):
    sub = feat[docs]
    labs = labels[docs]
    n = float(len(docs))
    c1_tot = float(np.sum(labs == 1))
    h_par = ent(c1_tot, n - c1_tot)

    is_c1 = (labs == 1)
    left_n = sub.sum(axis=0).astype(float)
    right_n = n - left_n
    left_c1 = sub[is_c1].sum(axis=0).astype(float)
    right_c1 = c1_tot - left_c1

    h_left = ent_vec(left_c1, left_n - left_c1)
    h_right = ent_vec(right_c1, right_n - right_c1)

    if method == 1:
        ig = h_par - 0.5 * h_left - 0.5 * h_right
    else:
        ig = h_par - (left_n / n) * h_left - (right_n / n) * h_right

    best = int(np.argmax(ig))
    return best, float(ig[best])


# === vectorized tree prediction ===
def calc_acc(root, feat, labels):
    preds = np.zeros(len(labels), dtype=int)
    stack = [(root, np.arange(len(labels)))]
    while stack:
        node, idxs = stack.pop()
        if node['type'] == 'leaf':
            preds[idxs] = node['pred']
        else:
            mask = feat[idxs, node['word']] == 1
            if np.any(mask):
                stack.append((node['left'], idxs[mask]))
            if np.any(~mask):
                stack.append((node['right'], idxs[~mask]))
    return np.mean(preds == labels) * 100


# === tree display ===
def show_tree(root, words):
    stack = [(root, 0, None)]
    while stack:
        node, depth, label = stack.pop()
        pad = "    " * depth
        if label:
            print(f"{'    ' * (depth - 1)}  {label}")
        if node['type'] == 'leaf':
            tag = 'atheism' if node['pred'] == 1 else 'books'
            print(f"{pad}[{tag}] ({len(node['docs'])} docs)")
        else:
            name = words[node['word']]
            print(f"{pad}{name} (IG={node['ig']:.6f})")
            stack.append((node['right'], depth + 1, 'no:'))
            stack.append((node['left'], depth + 1, 'yes:'))


# === build tree with iterative priority queue ===
def build_tree(feat, labels, t_feat, t_lab, words, method, max_nodes=100):
    counter = [0]

    def make_leaf(docs, fallback=1):
        idx = counter[0]
        counter[0] += 1
        if len(docs) == 0:
            pred = fallback
        else:
            c1 = int(np.sum(labels[docs] == 1))
            pred = 1 if c1 >= len(docs) - c1 else 2
        return {'type': 'leaf', 'docs': docs, 'idx': idx, 'pred': pred}

    root = make_leaf(np.arange(len(labels)))
    bw, big = find_best(feat, labels, root['docs'], method)

    heap = [(-big, root['idx'])]
    table = {root['idx']: (root, bw, big)}
    train_accs, test_accs = [], []

    for step in range(max_nodes):
        if not heap:
            break

        _, key = heapq.heappop(heap)
        leaf, word, ig = table.pop(key)

        docs = leaf['docs']
        mask = feat[docs, word] == 1
        left = make_leaf(docs[mask], leaf['pred'])
        right = make_leaf(docs[~mask], leaf['pred'])

        leaf['type'] = 'internal'
        leaf['word'] = word
        leaf['ig'] = ig
        leaf['left'] = left
        leaf['right'] = right

        for child in [left, right]:
            if len(child['docs']) > 0:
                bw, big = find_best(feat, labels, child['docs'], method)
                heapq.heappush(heap, (-big, child['idx']))
                table[child['idx']] = (child, bw, big)

        train_accs.append(calc_acc(root, feat, labels))
        test_accs.append(calc_acc(root, t_feat, t_lab))

        if step == 9:
            print(f"\nTree after 10 nodes (Method {method}):")
            show_tree(root, words)

    return root, train_accs, test_accs


# === main ===
if __name__ == '__main__':
    DATA = "a3_q1/a3_q1"
    train_feat, train_lab, words = load_bow(
        f"{DATA}/trainData.txt", f"{DATA}/trainLabel.txt", f"{DATA}/words.txt")
    test_feat, test_lab, _ = load_bow(
        f"{DATA}/testData.txt", f"{DATA}/testLabel.txt", f"{DATA}/words.txt")

    for method in [1, 2]:
        print(f"\n{'=' * 60}")
        print(f"Method {method}")
        print('=' * 60)

        root, tr_acc, te_acc = build_tree(
            train_feat, train_lab, test_feat, test_lab, words, method)

        xs = np.arange(1, len(tr_acc) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(xs, tr_acc, label='Train')
        plt.plot(xs, te_acc, label='Test')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Decision Tree Accuracy (Method {method})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'workworkspace/q1_method{method}_accuracy.png', dpi=150)
        plt.close()

        print(f"\nFinal acc (100 nodes) - Train: {tr_acc[-1]:.2f}%, Test: {te_acc[-1]:.2f}%")
