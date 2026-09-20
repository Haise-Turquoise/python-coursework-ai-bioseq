import sys
import code

def run_q2():
    rew = -1.0
    final_utils, hist = code.value_iter(rew)
    pol = code.get_policy(final_utils, rew)
    code.print_results(final_utils, pol, hist, rew)

    print('\n\n' + '=' * 60)
    print('Q2(b): Policy changes for R(s) = -1.6 to -0.5')
    print('=' * 60)
    for r_val in [-1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5]:
        u, h = code.value_iter(r_val)
        p = code.get_policy(u, r_val)
        print(f'\nR(s) = {r_val}')
        print(f'       col1   col2   col3   col4')
        for row in range(3):
            cells = [f"{p[row][col]:^6}" for col in range(4)]
            print(f"row{row+1}  {'  '.join(cells)}")

    print('\n\n' + '=' * 60)
    print('Q2(c): Policy changes for R(s) = -0.08 to -0.03')
    print('=' * 60)
    for r_val in [-0.08, -0.07, -0.06, -0.05, -0.04, -0.03]:
        u, h = code.value_iter(r_val)
        p = code.get_policy(u, r_val)
        print(f'\nR(s) = {r_val}')
        print(f'       col1   col2   col3   col4')
        for row in range(3):
            cells = [f"{p[row][col]:^6}" for col in range(4)]
            print(f"row{row+1}  {'  '.join(cells)}")


def run_q3():
    vit_path, vit_delta, vit_bp = code.viterbi(code.OBS)
    code.print_viterbi(vit_path, vit_delta, vit_bp)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode == 'q2':
        run_q2()
    elif mode == 'q3':
        run_q3()
    else:
        run_q2()
        print('\n\n')
        run_q3()
