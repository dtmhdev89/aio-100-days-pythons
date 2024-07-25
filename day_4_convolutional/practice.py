"""
output_height = ((H + 2P - K) / S) + 1
output_width = ((W + 2P - K) / S) + 1

số lần lướt kernel trên ảnh đầu vào sẽ là output_height x output_width

H: Chiều cao của ảnh đầu vào.
W: Chiều rộng của ảnh đầu vào.
K: Kích thước kernel (thường là số lẻ, ví dụ như 3, 5, 7,...).
S: Bước nhảy (stride), là số lượng pixel mà kernel di chuyển mỗi lần.
P: Đệm (padding), là số lượng pixel được thêm vào xung quanh ảnh đầu vào.
"""

def compute_output_dimension(H, K, S, P):
    return ((H + 2 * P - K) / S) + 1

def flatten_2d_matrix(matrix):
    return tuple([num for row in matrix for num in row])

def compute_convolutional(matrix1, matrix2):
    return sum(num1 * num2 for num1, num2 in zip(matrix1, matrix2))

def compute_slicing_kernel_indices(H, K):
    vector = list()
    
    for i in range(H):
        j = i + 1

        while j < H:
            distance = j - i + 1
            if distance == K: vector.append(tuple([i, j]))

            j += 1

    return vector

def convolutional_2d_calculator(matrix, kernel):
    if (len(kernel) == 0) or (len(matrix) == 0): return None

    H = len(matrix)
    W = len(matrix[0])
    K_h = len(kernel)
    K_w = len(kernel[0])
    # S = 1
    # P = 0

    # output_height = compute_output_dimension(H, K_h, S, P)
    # output_width = compute_output_dimension(W, K_w, S, P)
    # slicing_num = output_height * output_width

    h_vectors = compute_slicing_kernel_indices(H, K_h)
    w_vectors = compute_slicing_kernel_indices(W, K_w)

    convolutional_matrix = list()
    flatten_kernel = flatten_2d_matrix(kernel)
    
    for h_min, h_max in h_vectors:
        convolutional_matrix.append(list())

        for w_min, w_max in w_vectors:
            slicing_kernel_matrix = tuple([num for row in matrix[h_min:h_max + 1] for num in row[w_min:w_max + 1]])
            convolutional_matrix[-1].append(compute_convolutional(slicing_kernel_matrix, flatten_kernel))

    return convolutional_matrix

def main():
    matrix_A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    kernel_B = [
        [2, 4],
        [1, 3]
    ]

    # C1
    print(convolutional_2d_calculator(matrix_A, kernel_B))

    # C2
    kernel_C = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ]

    print(convolutional_2d_calculator(matrix_A, kernel_C))

    # C3

    matrix_D = [
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [7, 8, 9, 10]
    ]

    print(convolutional_2d_calculator(matrix_D, kernel_B))
    print(convolutional_2d_calculator(matrix_D, kernel_C))


if __name__ == "__main__":
    main()
