
Calculator using python

def calculator():
    print("Simple Calculator")
    try:
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))
    except ValueError:
        print("Invalid input! Please enter numeric values.")
        return
    print("\nChoose an operation:")
    print("1. Addition (+)")
    print("2. Subtraction (-)")
    print("3. Multiplication (*)")
    print("4. Division (/)")
    operation = input("Enter the operation (1/2/3/4 or +, -, *, /): ")
    if operation == '1' or operation == '+':
        result = num1 + num2
        op_symbol = '+'
    elif operation == '2' or operation == '-':
        result = num1 - num2
        op_symbol = '-'
    elif operation == '3' or operation == '*':
        result = num1 * num2
        op_symbol = '*'
    elif operation == '4' or operation == '/':
        if num2 == 0:
            print("Error: Division by zero is not allowed.")
            return
        result = num1 / num2
        op_symbol = '/'
    else:
        print("Invalid operation selection.")
        return
    print(f"\nResult: {num1} {op_symbol} {num2} = {result}")
calculator()

 
