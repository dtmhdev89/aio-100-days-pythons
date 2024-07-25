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

def compute_compress_convolutional(matrix, mode):
    if mode == "Max Pooling":
        return max(matrix)
    elif mode == "Average Pooling":
        return sum(matrix) / len(matrix)

def add_zero_padding(matrix):
    H, W = len(matrix), len(matrix[0])
    w_zero_vector = [0] * (W + 2)
    deep_copied_matrix = [row[:] for row in matrix] 

    for i in range(H):
        deep_copied_matrix[i].insert(W, 0)
        deep_copied_matrix[i].insert(0, 0)

    deep_copied_matrix.insert(0, w_zero_vector)
    deep_copied_matrix.insert(H + 1, w_zero_vector)
    
    return deep_copied_matrix

def compute_slicing_kernel_indices(dim, pool_size_dim, S):
    vector = list()
    
    start_indices = list(range(0, dim - pool_size_dim + 1, S))
    end_indices = list(range(pool_size_dim - 1, dim + 1, S))

    vector.extend([tuple([i, j]) for i, j in zip(start_indices, end_indices)])

    return vector

def convolutional_2d_compressor(matrix, pool_size, S, mode):
    compute_matrix = matrix

    if (len(compute_matrix) == 0): return None

    H = len(compute_matrix)
    W = len(compute_matrix[0])
    Ps_h, Ps_w = pool_size
    # P = 0

    # output_height = compute_output_dimension(H, K_h, S, P)
    # output_width = compute_output_dimension(W, K_w, S, P)
    # slicing_num = output_height * output_width

    h_vectors = compute_slicing_kernel_indices(H, Ps_h, S)
    w_vectors = compute_slicing_kernel_indices(W, Ps_w, S)

    convolutional_matrix = list()
    
    for h_min, h_max in h_vectors:
        convolutional_matrix.append(list())

        for w_min, w_max in w_vectors:
            slicing_kernel_matrix = tuple([num for row in compute_matrix[h_min:h_max + 1] for num in row[w_min:w_max + 1]])
            convolutional_matrix[-1].append(compute_compress_convolutional(slicing_kernel_matrix, mode))

    return convolutional_matrix

def main():
    # C1
    matrix_A = [
        [0, 0, 0, 4],
        [0, 4, 0, 2],
        [0, 1, 0, 2],
        [0, 3, 0, 2]
    ]

    pool_size = (2, 2)
    S = 2

    print(convolutional_2d_compressor(matrix_A, pool_size, S, mode="Max Pooling"))
    print(convolutional_2d_compressor(matrix_A, pool_size, S, mode="Average Pooling"))

    # C2
    matrix_A = [
        [0, 0, 0],
        [0, 4, 0],
        [0, 1, 0]
    ]

    print(convolutional_2d_compressor(matrix_A, pool_size, S, mode="Max Pooling"))
    print(convolutional_2d_compressor(matrix_A, pool_size, S, mode="Average Pooling"))


    # C3

    matrix_D = [
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [7, 8, 9, 10]
    ]

    print(convolutional_2d_compressor(matrix_D, pool_size, S, mode="Max Pooling"))
    print(convolutional_2d_compressor(matrix_D, pool_size, S, mode="Average Pooling"))

if __name__ == "__main__":
    main()
