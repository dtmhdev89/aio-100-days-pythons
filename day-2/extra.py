def is_armstrong(a_number):
    a_number = int(a_number)
    if a_number < 18: print("Number should be greater than or equal to 18")

    a_number_str = str(a_number)

    for i in a_number_str:
        a_number -= int(i)**3
    
    return a_number == 0

def max_value_in_lst(lst_data):
    if len(lst_data) == 1: return lst_data[0]

    max_value = lst_data[0]
    for i in lst_data[1:]:
        if i > max_value: max_value = i
    
    return max_value

def extra_candies(lst_data, extra_candy):
    max_value = max_value_in_lst(lst_data)

    for i in range(len(lst_data)):
        lst_data[i] = (lst_data[i] + extra_candy) >= max_value
    
    return lst_data

def median_calculator(lst_data):
    sorted_lst_data = list(set(lst_data))
    n = len(sorted_lst_data)
    median_number = None

    if n % 2 == 0:
        mid_position = n // 2
        median_number = (sorted_lst_data[mid_position - 1] + sorted_lst_data[mid_position]) / 2
    else:
        mid_position = (n + 1) // 2
        median_number = sorted_lst_data[mid_position - 1]

    return median_number

def mean_calculator(lst_data):
    n = len(lst_data)

    if n == 0: return None

    sum_value = 0

    for i in lst_data:
        sum_value += i

    return sum_value / n

def bag_of_words_builder(corpus, case_sensitive=False):
    bag_of_words = list()

    for setences in corpus:
        words = setences.split(' ')
        
        for word in words:
            if word in bag_of_words: continue

            if case_sensitive:
                bag_of_words.append(word)
            else:
                bag_of_words.append(word.lower())

    bag_of_words.sort()
    return bag_of_words

def bag_of_words_based_vectorize(bag_of_words, a_sentence):
    word_lst = a_sentence.split(' ')
    vector = [0] * len(bag_of_words)

    for i in range(len(bag_of_words)):
        vector[i] = word_lst.count(bag_of_words[i])
    
    return vector

def consecutive_index_search(lst_data, search_word, multiple=False):
    index_result = list()

    for i in range(len(lst_data)):
        if lst_data[i] == search_word:
            index_result.append(i)

            if not multiple: break
    
    return index_result

def nearest_not_none_index_to_the_right(lst_data, start_index):
    idx_result = -1

    for i in range(start_index, len(lst_data)):
        if lst_data[i] != None:
            idx_result = i
            break
    
    return idx_result

def nearest_neighbor_interpolation(lst_data):
    for i in range(len(lst_data)):
        if lst_data[i] != None: continue

        if i == 0:
            lst_data[i] = nearest_not_none_index_to_the_right(lst_data, i + 1)
        else:
            lst_data[i] = lst_data[i - 1]
    
    return lst_data

# # Define the method
# def print_elements(self):
#     for element in self:
#         print(element)

# # Bind the method to the list class
# list.print_elements = print_elements
# This way won't work

class MatrixList(list):
    def mtx_plus(self, addedMatrixList):
        new_matrix = MatrixList()

        for i in range(len(self)):
            new_matrix.append([])

            for j in range(len(self[i])):
                new_matrix[i].append(self[i][j] + addedMatrixList[i][j])

        return new_matrix
    
    def mtx_subtract(self, subtractedMatrixList):
        new_matrix = MatrixList()

        for i in range(len(self)):
            new_matrix.append([])

            for j in range(len(self[i])):
                new_matrix[i].append(self[i][j] - subtractedMatrixList[i][j])

        return new_matrix

    def mtx_dot_product(self, productedMatrixList):
        new_matrix = MatrixList()

        for row_i in self:
            dot_product_row = list()

            for row_j in productedMatrixList.transpose():
                row_sum = 0
                
                for j in range(len(row_j)):
                    row_sum += (row_i[j] * row_j[j])

                dot_product_row.append(row_sum)
            
            new_matrix.append(dot_product_row)

        return new_matrix

    def transpose(self):
        new_matrix = MatrixList()

        for col_i in range(len(self[0])):
            transposed_row = list()

            for row in self:
                transposed_row.append(row[col_i])
            
            new_matrix.append(transposed_row)
        
        return new_matrix
    
class VectorTuple(tuple):
    def tpl_plus(self, addedVector):
        new_tuple = list()

        for i in range(len(self)):
            new_tuple.append(self[i] + addedVector[i])

        return tuple(new_tuple)

    def tpl_subtract(self, subtractedVector):
        new_tuple = list()

        for i in range(len(self)):
            new_tuple.append(self[i] - subtractedVector[i])
            
        return tuple(new_tuple)
    
    def tpl_multiply(self, multiplieddVector):
        new_tuple = list()

        for i in range(len(self)):
            new_tuple.append(self[i] * multiplieddVector[i])
            
        return tuple(new_tuple)
    
    def tpl_distance(self, toVector):
        normal_distance = sum((self[i] - toVector[i])**2 for i in range(len(self)))

        return VectorTuple.square_root(normal_distance)

    @staticmethod
    def square_root(number, precision=0.00001):
        # Newton method
        if number < 0:
            return None
        
        guess = number

        while abs(guess * guess - number) > precision:
            guess = (guess + number / guess) / 2

        return guess

def main():
    # BT 1
    print("-" * 20 + 'BT1')
    lst_data = list(range(1, 11))

    # First 5 elements
    print(lst_data[:5])

    # Odd elements
    result = ''
    sum_value = 0
    for i in lst_data:
        sum_value += i

        if i % 2 != 0: result += str(i) + ', '
    
    print(result[:-2])
    print(sum_value)

    # BT2
    print("-" * 20 + 'BT2')
    lst_data = list(range(2, 13, 2))

    print(lst_data)
    new_lst = []

    for i in lst_data:
        if i % 3 != 0: new_lst.append(i)

    lst_data = new_lst
    print(lst_data)

    for i in range(1, 4):
        lst_data.append(i)
    
    print(lst_data)

    counter = 3
    for i in range(6, 9):
        lst_data.insert(counter, i)
        counter += 1
    
    print(lst_data)

    counter = 0
    for i in lst_data:
        if (i % 2 == 0) or (i % 5 == 0): lst_data[counter] = 0
        counter += 1

    print(lst_data)

    # BT 3
    print("-" * 20 + 'BT3')

    test_case = [130, 270, 153, 407, 177, 371, 1000, 1634, 370]
    armstrong_lst = []

    for i in test_case:
        if is_armstrong(i): armstrong_lst.append(i)

    print(armstrong_lst)

    test_case_1 = [2, 3, 5, 1, 3]
    extra_candy = 3
    print(extra_candies(test_case_1, extra_candy))

    test_case_1 = [4, 2, 1, 1, 2]
    extra_candy = 1
    print(extra_candies(test_case_1, extra_candy))

    test_case_1 = [12, 1, 12]
    extra_candy = 10
    print(extra_candies(test_case_1, extra_candy))

    # BT 5
    print("-" * 20 + 'BT5')

    lst_data = list(range(1, 11))
    print(lst_data)
    print(median_calculator(lst_data))

    lst_odd = []
    for i in lst_data:
        if i % 2 != 0: lst_odd.append(i)
    
    lst_odd.sort(reverse=True)
    print(lst_odd)

    # BT 6
    print("-" * 20 + 'BT6')
    lst_data = list(range(1, 11))
    odd_lst = []
    even_lst = []

    for i in lst_data:
        if i % 2 == 0:
            even_lst.append(i)
        else:
            odd_lst.append(i)
    
    print(mean_calculator(odd_lst))
    print(mean_calculator(even_lst))

    print(mean_calculator(lst_data))
    print(median_calculator(lst_data))

    # BT7
    print("-" * 20 + 'BT7')

    corpus = ["Tôi thích môn Toán", "Tôi thích AI", "Tôi thích âm nhạc"]
    bag_of_words = bag_of_words_builder(corpus, case_sensitive=True)
    print(bag_of_words)
    a_str = "Tôi thích AI thích Toán"
    print(bag_of_words_based_vectorize(bag_of_words, a_str))

    # BT8
    print("-" * 20 + 'BT8')

    lst_data = [1, 1.1, None, 1.4, None, 1.5, None, 2.0]
    print(consecutive_index_search(lst_data, None))
    print(consecutive_index_search(lst_data, None, multiple=True))

    # BT9
    print("-" * 20 + 'BT9')
    lst_data = [1, 1.1, None, 1.4, None, 1.5, None, 2.0]
    print(nearest_neighbor_interpolation(lst_data))

    # BT10
    print("-" * 20 + 'BT10')
    rows, cols = 3, 3
    lst_2d_data = [0] * rows
    filled_values = list(range(1, 10))

    for row_i in range(rows):
        lst_2d_data[row_i] = [0] * cols

        for col_j in range(cols):
            lst_2d_data[row_i][col_j] = filled_values[rows * row_i + col_j]
    
    print(lst_2d_data)

    lst_2d_data = list()
    filled_values = list(range(1, 10))

    for row_i in range(rows):
        row_lst = list()

        for col_j in range(cols):
            row_lst.append(rows * row_i + col_j + 1)
        
        lst_2d_data.append(row_lst)

    print(lst_2d_data)

    sub_lst_2d = list()

    for row_i in range(len(lst_2d_data)):
        sub_row_lst = list()

        for col_j in range(len(lst_2d_data[row_i])):
            if (col_j == 0) or (col_j == 2): sub_row_lst.append(lst_2d_data[row_i][col_j])
    
        sub_lst_2d.append(sub_row_lst)
    
    print(sub_lst_2d)

    # BT 11
    print("-" * 20 + 'BT11')

    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    B = [
        [2, 4, 6],
        [1, 3, 5],
        [1, 0, 1]
    ]

    C = [
        [2, 4],
        [1, 3],
        [1, 0]
    ]

    matrix_A = MatrixList(A)
    matrix_B = MatrixList(B)
    print(matrix_A.mtx_plus(matrix_B))
    print(matrix_A.mtx_subtract(matrix_B))
    print(matrix_A.mtx_dot_product(matrix_B))

    matrix_C = MatrixList(C)
    print(matrix_A.mtx_dot_product(matrix_C))

    # BT 12
    # List Comprehension
    stop_words = ["I", "love", "and", "to"]
    input_sentence = "I love AI and listen to music"

    removed_stop_words_lst = [word for word in input_sentence.split(' ') if word not in stop_words]
    print(removed_stop_words_lst)

    # BT 13
    # List and Tuple
    tuple1 = (2,3)
    tuple2 = (3,6)

    vector1 = VectorTuple(tuple1)
    vector2 = VectorTuple(tuple2)
    print(vector1.tpl_plus(vector2))
    print(vector1.tpl_multiply(vector2))
    print(vector1.tpl_distance(vector2))

    print(tuple1.index(3))
    print(tuple2.index(3))

if __name__ == "__main__":
    main()
