class VectorTuple(tuple):
    def tpl_dot_product(self, another_vector):
        new_list = list()

        for i in range(len(self)):
            new_list.append(self[i] * another_vector[i])

        return sum(new_list)
        

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

def compute_cosine_similarity(vector_A, vector_B):
    """
    Perform cosine similarity between two vector

    Parameter:
    vectorA: list
    vectorB: list

    Two vector must be same size

    Returns:
    float: value of cosine similarity
    """

    vector_0 = VectorTuple([0, 0])

    dot_product = vector_A.tpl_dot_product(vector_B)
    result = dot_product / (vector_A.tpl_distance(vector_0) * vector_B.tpl_distance(vector_0))

    return result

def main():
    vector_A = VectorTuple([1, 2])
    vector_B = VectorTuple([4, 5])

    print(compute_cosine_similarity(vector_A, vector_B))

if __name__ == "__main__":
    main()
