CODSOFT Task 1
Chatbot with Rule Based Responses

print("Chatbot: Hello! I am a simple rule-based chatbot.")
print("Type 'bye' to exit.\n")

while True:
    user_input = input("You: ").lower()

    if "hello" in user_input or "hi" in user_input:
        print("Chatbot: Hello! How can I help you today?")
    
    elif "your name" in user_input:
        print("Chatbot: I am a rule-based chatbot created using Python.")
    
    elif "how are you" in user_input:
        print("Chatbot: I'm just a program, but I'm working perfectly!")
    
    elif "weather" in user_input:
        print("Chatbot: I cannot check weather, but I hope it's nice where you are!")
    
    elif "bye" in user_input:
        print("Chatbot: Goodbye! Have a great day!")
        break

    else:
        print("Chatbot: Sorry, I don't understand that. Try asking something else.")


CODSOFT TASK 2
TIC TAC TOE AI

import math
board = [" " for _ in range(9)]

def print_board():
    print()
    for i in range(3):
        print(" ", board[3*i], "|", board[3*i+1], "|", board[3*i+2])
        if i < 2:
            print("---|---|---")
    print()
def check_winner(brd, player):
    win_conditions = [
        [0,1,2], [3,4,5], [6,7,8],  # rows
        [0,3,6], [1,4,7], [2,5,8],  # columns
        [0,4,8], [2,4,6]            # diagonals
    ]
    for combo in win_conditions:
        if brd[combo[0]] == brd[combo[1]] == brd[combo[2]] == player:
            return True
    return False
def is_draw(brd):
    return " " not in brd
def minimax(brd, depth, is_maximizing):
    if check_winner(brd, "O"):
        return 1
    if check_winner(brd, "X"):
        return -1
    if is_draw(brd):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if brd[i] == " ":
                brd[i] = "O"
                score = minimax(brd, depth + 1, False)
                brd[i] = " "
                best_score = max(best_score, score)
        return best_score

    else:
        best_score = math.inf
        for i in range(9):
            if brd[i] == " ":
                brd[i] = "X"
                score = minimax(brd, depth + 1, True)
                brd[i] = " "
                best_score = min(best_score, score)
        return best_score
def ai_move():
    best_score = -math.inf
    best_move = None

    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(board, 0, False)
            board[i] = " "
            if score > best_score:
                best_score = score
                best_move = i

    board[best_move] = "O"
def human_move():
    while True:
        pos = int(input("Enter your move (1-9): ")) - 1
        if 0 <= pos <= 8 and board[pos] == " ":
            board[pos] = "X"
            break
        else:
            print("Invalid move, try again.")


CODSOFT TASK 2
def play():
    print("Welcome to Tic-Tac-Toe!")
    print("You are X, AI is O.")
    print_board()

    while True:
        human_move()
        print_board()

        if check_winner(board, "X"):
            print("You win!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

        print("AI is thinking...")
        ai_move()
        print_board()

            print("It's a draw!")
            break
play()



CODSOFT TASK 3