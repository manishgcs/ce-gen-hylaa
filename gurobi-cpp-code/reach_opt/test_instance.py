def test_vals():
    with open("test_instance_vals") as input_file:
        for line in input_file:
            print(line)
            line = line.strip()
            numbers = line.split()
            total_val = 0.0
            val1 = None
            val2 = None
            for number in numbers:
                if number is not "+" and val1 is None:
                    val1 = float(number)
                elif number is not "+" and val1 is not None:
                    val2 = float(number)
                elif number is "+":
                    if val2 is not None:
                        total_val += val1 * val2
                    else:
                        total_val += val1
                    val1 = val2 = None
            if val2 is not None:
                total_val += val2*val1
            else:
                total_val += val1
            val1 = val2 = None
            print(total_val)
            print("\n")

if __name__ == '_main()__':
    print("Hello")
    test_vals()
