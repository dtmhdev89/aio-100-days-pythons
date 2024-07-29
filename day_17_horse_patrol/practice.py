import numpy as np

"""
Định lí Schwenk: Cho bàn cờ m × n bất kỳ với m ≤ n. Không có hành trình đóng nào của quân mã nếu một trong ba điều kiện dưới đây xảy ra:
m và n đều lẻ.
m = 1, 2, hoặc 4; n khác 1.
m = 3 và n = 4, 6 hoặc 8.
"""

def build_possible_positions(N):
    possisions = dict()

    for i in range(N):
        for j in range(N):
            possisions[tuple([i, j])] = possible_positions(tuple([i, j]), N).tolist()
    
    return possisions

def possible_positions(current_pos, N):
    row_i, col_j = current_pos
    possible_steps = np.array([
        (row_i + 1, col_j + 2),
        (row_i + 1, col_j - 2),
        (row_i - 1, col_j + 2),
        (row_i - 1, col_j - 2),
        (row_i + 2, col_j + 1),
        (row_i - 2, col_j + 1),
        (row_i + 2, col_j - 1),
        (row_i - 2, col_j - 1),
    ])

    return possible_steps[(np.min(possible_steps, axis=1) >= 0) & (np.max(possible_steps, axis=1) <= (N - 1))]

def main():
    N = int(input("Input size of chessboard: "))

    board = np.arange(1, N**2 + 1).reshape(N, N)

    # Nhap vi tri bat dau
    print("Input horse's position: ")
    x = int(input("X: "))
    y = int(input("Y: "))
    current_pos = np.array([x, y])

    refer_positions = build_possible_positions(N)

    possible_steps = build_possible_positions(N)

    made_steps = np.array([current_pos])

    early_stop = 0

    marked_board = list()

    counter = 0

    while (len(made_steps) < N**2) & (early_stop < 20000000):
        early_stop += 1
        counter += 1
        # print(f'**current step: {tuple(made_steps[-1])}')
        current_pos_steps = possible_steps[tuple(made_steps[-1])]
        if len(current_pos_steps) > 0:
            next_step = possible_steps[tuple(made_steps[-1])].pop(0)
            
            # print('++stack')
            # print(possible_steps[tuple(made_steps[-1])])
            # print(f"step: {next_step}")

            if np.any(np.all(made_steps == next_step, axis=1)):
                # print("--> cont")
                continue
            
            made_steps = np.vstack([made_steps, next_step])
            # print("--add")
            # print(made_steps)
        else:
            # print("--back")
            stuck_step = made_steps[-1]
            possible_steps[tuple(stuck_step)] = np.copy(refer_positions[tuple(stuck_step)]).tolist()
            made_steps = made_steps[:-1, :]
            # print(made_steps)
    
    for row in made_steps:
        marked_board.append(board[row[0], row[1]])

    print(f"steps:\n {np.array(marked_board)}")
    print(len(marked_board))
    print(f"board:\n{board}")
    print(f"run counter: {counter}")

if __name__ == "__main__":
    main()
