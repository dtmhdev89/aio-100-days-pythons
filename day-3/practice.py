def main():
    def find_square_root(number):
        """Find square root of a number"""
        if number < 0: return None

        EPSILON = 0.001
        x_n = number
        n = 0

        x_n1 = improve_x_n(x_n, number)

        while abs(x_n1 - x_n) >= EPSILON:
            n += 1
            x_n = x_n1
            x_n1 = improve_x_n(x_n, number)

        return x_n1
    
    def fx_function(x, number):
        return x**2 - number
    
    def gradient_descent_fx(x):
        return 2 * x
    
    def improve_x_n(x_n, number):
        return x_n - (fx_function(x_n, number) / gradient_descent_fx(x_n))

    # test case
    print(find_square_root(2))
    print(find_square_root(3))

if __name__ == "__main__":
    main()
