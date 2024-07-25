class Staff():
    def __init__(self, name, age, address, salary, total_time) -> None:
        self.name = name
        self.age = age
        self.address = address
        self.salary = salary
        self.total_time = total_time
    
    def show_info(self):
        return f"name: {self.name}| age: {self.age} | address: {self.address} | salary: {self.salary} | total_time: {self.total_time} "

    def calculate_bonus(self):
        total_time_condition = [
            self.total_time >= 200,
            (self.total_time >= 100 and self.total_time < 200),
            self.total_time < 100
        ]

        bonus_amount = [
            self.salary * 0.2,
            self.salary * 0.1,
            0
        ]

        return sum(cond * bonus for cond, bonus in zip(total_time_condition, bonus_amount))

def main():
    staff_A = Staff(name="A", age=30, address="Galaxy 1", salary=30, total_time=300)
    staff_B = Staff(name="B", age=30, address="Galaxy 2", salary=50, total_time=199)
    staff_C = Staff(name="C", age=30, address="Galaxy 3", salary=10, total_time=99)

    print(staff_A.calculate_bonus())
    print(staff_B.calculate_bonus())
    print(staff_C.calculate_bonus())

if __name__ == "__main__":
    main()
