CAN_SETTING = ['Canh', 'Tan', 'Nham', 'Quy', 'Giap', 'At', 'Binh', 'Dinh', 'Mau', 'Ky']
CHI_SETTING = ['Than', 'Dau', 'Tuat', 'Hoi', 'Ty', 'Suu', 'Dan', 'Meo', 'Thin', 'Ty', 'Ngo', 'Mui']

def calculate_can_chi_calendar(year):
    year = int(year)

    result = ' '.join([CAN_SETTING[year % 10], CHI_SETTING[year % 12]])

    return result;

def main():
    # Test case 1
    year = 2024
    print(calculate_can_chi_calendar(year))

    # Test case 2
    year = 2023
    print(calculate_can_chi_calendar(year))

    # Test case 3
    year = 1997
    print(calculate_can_chi_calendar(year))

if __name__ == "__main__":
    main()
