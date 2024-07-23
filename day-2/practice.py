def compute_interest(money, period):
    money = float(money)
    period = int(period) # in day

    result = round((money + (money / period)) ** period, 3)
    
    return result

def main():
    # Test case 1
    money, period = 1, 12
    print(compute_interest(money, period))
    
    # Test case 2
    money, period = 1, 365
    print(compute_interest(money, period))

    # Test case 3
    money, period = 1, 730
    print(compute_interest(money, period))

if __name__ == "__main__":
    main()
