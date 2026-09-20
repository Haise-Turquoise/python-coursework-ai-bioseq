import numpy as np

# === Grid World Config ===
NROWS = 3
NCOLS = 4
WALL = (1, 1)          # s(2,2) in 1-indexed = (1,1) in 0-indexed
TERMS = {(1, 3): -1.0, (2, 3): 1.0}  # terminal states: s(2,4)=-1, s(3,4)=+1
GAMMA = 1.0

# actions: (dr, dc), and their left/right perpendicular
ACTS = {
    'UP':    (-1, 0),
    'DOWN':  (1, 0),
    'LEFT':  (0, -1),
    'RIGHT': (0, 1),
}
ACT_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']

PERP = {
    'UP':    ('LEFT', 'RIGHT'),
    'DOWN':  ('RIGHT', 'LEFT'),
    'LEFT':  ('DOWN', 'UP'),
    'RIGHT': ('UP', 'DOWN'),
}


def move(row, col, action):
    dr, dc = ACTS[action]
    nr, nc = row + dr, col + dc
    if nr < 0 or nr >= NROWS or nc < 0 or nc >= NCOLS:
        return (row, col)
    if (nr, nc) == WALL:
        return (row, col)
    return (nr, nc)


def value_iter(reward):
    utils = np.zeros((NROWS, NCOLS))

    # terminal values are fixed
    for (tr, tc), tv in TERMS.items():
        utils[tr][tc] = tv

    history = [utils.copy()]

    while True:
        new_utils = np.zeros((NROWS, NCOLS))

        for (tr, tc), tv in TERMS.items():
            new_utils[tr][tc] = tv

        for row in range(NROWS):
            for col in range(NCOLS):
                if (row, col) == WALL or (row, col) in TERMS:
                    continue

                best_val = float('-inf')
                for act in ACT_NAMES:
                    left_act, right_act = PERP[act]
                    s_fwd = move(row, col, act)
                    s_left = move(row, col, left_act)
                    s_right = move(row, col, right_act)

                    qval = (0.8 * utils[s_fwd[0]][s_fwd[1]]
                            + 0.1 * utils[s_left[0]][s_left[1]]
                            + 0.1 * utils[s_right[0]][s_right[1]])

                    if qval > best_val:
                        best_val = qval

                new_utils[row][col] = reward + GAMMA * best_val

        history.append(new_utils.copy())

        # convergence check: 3 decimal places
        if np.array_equal(np.round(new_utils, 3), np.round(utils, 3)):
            break

        utils = new_utils

    return new_utils, history


def get_policy(utils, reward):
    policy = [['' for _ in range(NCOLS)] for _ in range(NROWS)]
    arrows = {'UP': 'U', 'DOWN': 'D', 'LEFT': 'L', 'RIGHT': 'R'}

    for row in range(NROWS):
        for col in range(NCOLS):
            if (row, col) == WALL:
                policy[row][col] = 'X'
                continue
            if (row, col) in TERMS:
                policy[row][col] = f"{TERMS[(row, col)]:+.0f}"
                continue

            best_val = float('-inf')
            best_acts = []
            for act in ACT_NAMES:
                left_act, right_act = PERP[act]
                s_fwd = move(row, col, act)
                s_left = move(row, col, left_act)
                s_right = move(row, col, right_act)

                qval = (0.8 * utils[s_fwd[0]][s_fwd[1]]
                        + 0.1 * utils[s_left[0]][s_left[1]]
                        + 0.1 * utils[s_right[0]][s_right[1]])

                if round(qval, 3) > round(best_val, 3):
                    best_val = qval
                    best_acts = [act]
                elif round(qval, 3) == round(best_val, 3):
                    best_acts.append(act)

            policy[row][col] = ''.join(arrows[a] for a in best_acts)

    return policy


def print_results(utils, policy, history, reward):
    print("=" * 60)
    print(f"Value Iteration Results - R(s) = {reward}")
    print("=" * 60)

    # utility history
    for k, snap in enumerate(history):
        print(f"\nIteration {k}:")
        for row in range(NROWS):
            vals = []
            for col in range(NCOLS):
                if (row, col) == WALL:
                    vals.append("   X    ")
                else:
                    vals.append(f"{snap[row][col]:>8.3f}")
            print("  ".join(vals))

    # optimal policy
    print(f"\nOptimal Policy:")
    print(f"       col1   col2   col3   col4")
    for row in range(NROWS):
        cells = []
        for col in range(NCOLS):
            cells.append(f"{policy[row][col]:^6}")
        print(f"row{row+1}  {'  '.join(cells)}")


# === Q3 Bonus: Viterbi Decoding ===
PRIOR = {True: 0.4, False: 0.6}
TRANS = {True: {True: 0.7, False: 0.3}, False: {True: 0.2, False: 0.8}}
EMIT = {True: {True: 0.9, False: 0.1}, False: {True: 0.2, False: 0.8}}
OBS = [True, True, True, False]


def viterbi(obs):
    states = [True, False]
    nsteps = len(obs)

    delta = [{} for _ in range(nsteps)]
    backptr = [{} for _ in range(nsteps)]

    # base case
    for s in states:
        delta[0][s] = PRIOR[s] * EMIT[s][obs[0]]
        backptr[0][s] = None

    # recursive case
    for t in range(1, nsteps):
        for s in states:
            best_prob = -1
            best_prev = None
            for prev in states:
                prob = delta[t-1][prev] * TRANS[prev][s]
                if prob > best_prob:
                    best_prob = prob
                    best_prev = prev
            delta[t][s] = best_prob * EMIT[s][obs[t]]
            backptr[t][s] = best_prev

    # termination
    best_final = max(states, key=lambda s: delta[nsteps-1][s])

    # traceback
    path = [None] * nsteps
    path[nsteps-1] = best_final
    for t in range(nsteps-2, -1, -1):
        path[t] = backptr[t+1][path[t+1]]

    return path, delta, backptr


def print_viterbi(path, delta, backptr):
    print("=" * 60)
    print("Q3 Bonus: Viterbi Decoding")
    print("=" * 60)
    print(f"Observations: {OBS}")

    for t in range(len(OBS)):
        print(f"\nStep {t}:")
        for s in [True, False]:
            bp = backptr[t][s]
            bp_str = f"backptr={bp}" if bp is not None else "init"
            print(f"  S{t}={s:5}  delta={delta[t][s]:.6f}  {bp_str}")

    tag = {True: 'T', False: 'F'}
    seq = ', '.join(f"S{t}={tag[s]}" for t, s in enumerate(path))
    print(f"\nOptimal hidden state sequence: {seq}")
    print(f"  = ({', '.join(tag[s] for s in path)})")


if __name__ == '__main__':
    # === Q2(a): R(s) = -1.0 ===
    rew = -1.0
    final_utils, hist = value_iter(rew)
    pol = get_policy(final_utils, rew)
    print_results(final_utils, pol, hist, rew)

    # === Q2(b): R(s) from -1.6 to -0.5 ===
    print("\n\n" + "=" * 60)
    print("Q2(b): Policy changes for R(s) = -1.6 to -0.5")
    print("=" * 60)
    for r_val in [-1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5]:
        u, h = value_iter(r_val)
        p = get_policy(u, r_val)
        print(f"\nR(s) = {r_val}")
        print(f"       col1   col2   col3   col4")
        for row in range(NROWS):
            cells = [f"{p[row][col]:^6}" for col in range(NCOLS)]
            print(f"row{row+1}  {'  '.join(cells)}")

    # === Q2(c): R(s) from -0.08 to -0.03 ===
    print("\n\n" + "=" * 60)
    print("Q2(c): Policy changes for R(s) = -0.08 to -0.03")
    print("=" * 60)
    for r_val in [-0.08, -0.07, -0.06, -0.05, -0.04, -0.03]:
        u, h = value_iter(r_val)
        p = get_policy(u, r_val)
        print(f"\nR(s) = {r_val}")
        print(f"       col1   col2   col3   col4")
        for row in range(NROWS):
            cells = [f"{p[row][col]:^6}" for col in range(NCOLS)]
            print(f"row{row+1}  {'  '.join(cells)}")

    # === Q3 Bonus: Viterbi ===
    vit_path, vit_delta, vit_bp = viterbi(OBS)
    print("\n\n")
    print_viterbi(vit_path, vit_delta, vit_bp)
